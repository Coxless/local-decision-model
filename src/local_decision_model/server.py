"""ローカルサーバー: System One (TypeSafe の jev) の HTTP API と同じ形で回答を返す。

    POST /v1/systemone   状態 + 質問 → 回答
    GET  /v1/models      使えるモデルの名前
    GET  /health         状態確認 (モデルの読み込み中は 503)

リクエストとレスポンスの形は、公式 SDK (typesafe-sdk) が持つ OpenAPI のスキーマに合わせている。
検証エラー (422) も同じ形 (``{"detail": [{"loc", "msg", "type"}]}``) で返す。

- モデルは起動時に別スレッドで読み込む (OpenVINO はコンパイルもする)。終わるまでは 503 を返す
- 推論は 1 つずつ順番に行う (NPU や GPU を複数のリクエストで同時に使わない)
- 認証はしない (``Authorization`` ヘッダーは見ない)。localhost で待ち受ける前提
- score と、オブジェクトや配列で書いた instructions / criteria には未対応 (422 を返す)
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from .decide import Decider
from .schema import Question, parse_question, state_text

MODEL_PREFIX = "local-decision-model"
# jev のクラウドとベース URL だけで切り替えられるように、jev の名前も既定のモデルとして受け付ける
ALIAS_PREFIX = "jev-"
LISTED_ALIAS = "jev-latest"

Structured = dict[str, Any] | list[Any]


class NoulCriteria(BaseModel):
    true: str | Structured | None = None
    false: str | Structured | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str | Structured | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str | Structured | None = None
    criteria: dict[str, str | Structured | None]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str | Structured | None = None
    criteria: list[str | Structured] = Field(min_length=1)


QuestionSpec = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: str | Structured
    model: str
    questions: dict[str, QuestionSpec] = Field(min_length=1)


@dataclass(frozen=True)
class ModelInfo:
    """GET /v1/models で返すモデルの情報。"""

    name: str
    description: str
    release_date: str  # YYYY-MM-DD

    def accepts(self, requested: str) -> bool:
        return requested in (self.name, MODEL_PREFIX) or requested.startswith(ALIAS_PREFIX)


def model_name(model: str) -> str:
    """モデルディレクトリか HF ID から、API で使うモデルの名前を作る。"""
    return f"{MODEL_PREFIX}-{model.rstrip('/').split('/')[-1]}"


class _Invalid(Exception):
    def __init__(self, loc: list[str | int], msg: str, kind: str = "value_error"):
        self.error = {"type": kind, "loc": ["body", *loc], "msg": msg}


def _text(value: Any, loc: list[str | int]) -> str | None:
    """文字列か None だけを通す。オブジェクトや配列は未対応として検証エラーにする。"""
    if value is None or isinstance(value, str):
        return value
    raise _Invalid(loc, "オブジェクトや配列には未対応です。文字列で書いてください", "unsupported")


def _question(name: str, spec: NoulQuestion | ChoiceQuestion | ScoreQuestion) -> Question:
    loc: list[str | int] = ["questions", name, spec.type]
    if isinstance(spec, ScoreQuestion):
        raise _Invalid(loc, "score には未対応です", "unsupported")
    raw: dict[str, Any] = {
        "type": spec.type,
        "instructions": _text(spec.instructions, [*loc, "instructions"]),
    }
    if isinstance(spec, NoulQuestion):
        if spec.criteria:
            # criteria.false は使わない (仮説文は「真のとき成り立つ文」だけ)
            raw["criteria"] = {"true": _text(spec.criteria.true, [*loc, "criteria", "true"])}
    else:
        raw["criteria"] = {
            option: _text(description, [*loc, "criteria", option])
            for option, description in spec.criteria.items()
        }
    try:
        return parse_question(raw)
    except ValueError as e:
        raise _Invalid(loc, str(e)) from e


def to_questions(specs: dict[str, Any]) -> dict[str, Question]:
    """検証済みのリクエストの questions を質問にする。直せない点は、まとめて 422 にする。"""
    questions: dict[str, Question] = {}
    errors = []
    for name, spec in specs.items():
        try:
            questions[name] = _question(name, spec)
        except _Invalid as e:
            errors.append(e.error)
    if errors:
        raise RequestValidationError(errors)
    return questions


def count_tokens(tokenizer, state: str, questions: dict[str, Question]) -> int:
    """状態と仮説のトークン数 (usage.input_tokens)。切り詰めや特殊トークンは数えない。"""
    texts = [state] + [h for q in questions.values() for h in q.hypotheses()]
    return sum(len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"])


def create_app(load: Callable[[], Decider], model: ModelInfo) -> FastAPI:
    """load はモデルを読み込んで decider を返す。別スレッドで呼び、終わるまでは 503 を返す。

    読み込みの終わり (成功か失敗) は ``app.state.loaded`` (threading.Event) で待てる。
    """
    app = FastAPI(title="local-decision-model", docs_url=None, redoc_url=None)
    app.state.decider = None
    app.state.error = None
    app.state.loaded = threading.Event()
    lock = threading.Lock()  # 推論は 1 つずつ

    def run_load() -> None:
        try:
            app.state.decider = load()
        except Exception as e:  # 起動は続け、/health で原因を返す
            app.state.error = f"{type(e).__name__}: {e}"
        finally:
            app.state.loaded.set()

    threading.Thread(target=run_load, name="load-model", daemon=True).start()

    def ready_decider() -> Decider:
        if app.state.error:
            raise HTTPException(503, f"モデルを読み込めませんでした: {app.state.error}")
        if app.state.decider is None:
            raise HTTPException(503, "モデルを読み込んでいます", headers={"Retry-After": "1"})
        return app.state.decider

    @app.get("/health")
    def health() -> dict[str, str]:
        ready_decider()
        return {"status": "ok", "model": model.name}

    @app.get("/v1/models")
    def models() -> dict[str, list[dict[str, str]]]:
        names = [model.name, LISTED_ALIAS]
        return {
            "models": [
                {"name": n, "description": model.description, "release_date": model.release_date}
                for n in names
            ]
        }

    @app.post("/v1/systemone")
    def system_one(request: SystemOneRequest) -> dict[str, Any]:
        if not model.accepts(request.model):
            raise RequestValidationError(
                [
                    {
                        "type": "value_error",
                        "loc": ["body", "model"],
                        "msg": f"未知のモデルです: {request.model!r} ({model.name} か jev-*)",
                    }
                ]
            )
        questions = to_questions(request.questions)
        state = state_text(request.state)
        decider = ready_decider()
        try:
            with lock:
                answers = decider.decide(state, questions)
        except ValueError as e:  # 質問が長すぎて 1 回の推論に入らない、など
            raise RequestValidationError(
                [{"type": "value_error", "loc": ["body", "questions"], "msg": str(e)}]
            ) from e
        tokenizer = getattr(decider.scorer, "tokenizer", None)
        return {
            "model": model.name,
            "answers": {name: a.to_dict() for name, a in answers.items()},
            "usage": {
                "input_tokens": count_tokens(tokenizer, state, questions) if tokenizer else 0,
                "output_tokens": 0,
            },
        }

    return app
