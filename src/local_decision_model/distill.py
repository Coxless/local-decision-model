"""学習データの形式と、ペア方式のモデルによるラベル付け。

教師は Claude Code (scripts/phase2/label_with_claude.py)。ここの distill は、同じ形式のラベルを
ペア方式のモデルで付ける。教師のラベルと比べる基準に使う。

入力:
    状態        JSONL。1 行 = {"state": "..."}。行に "questions" があれば、質問セットの代わりに使う
    質問セット  YAML。questions の書き方は examples/*.yaml と同じ (state は要らない)

出力 (JSONL。1 行 = 1 つの状態 + 複数の質問):
    {"state": "...", "questions": {"refund": "...", "topic": {"type": "choice", ...}},
     "labels": {"refund": 0.94, "topic": {"配送": 0.81, "品質": 0.12, "その他": 0.07}}}

labels は、noul が確率、choice が選択肢 → 確率 (合計 1)。
教師のラベルも、人が付ける評価セットも、同じ形式で書く。
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .decide import Decider
from .schema import Choice, NoulAnswer, parse_questions

Label = float | dict[str, float]


@dataclass(frozen=True)
class Example:
    """学習データの行。questions は書いたままの形 (parse_questions に渡せる)。"""

    state: str
    questions: dict[str, Any]
    labels: dict[str, Label]

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "questions": self.questions, "labels": self.labels}


def load_question_set(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        specs = yaml.safe_load(f)["questions"]
    parse_questions(specs)  # 書き方の間違いを、推論を始める前に見つける
    return specs


def _rows(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                yield number, json.loads(line)


def load_states(
    path: str | Path, questions: dict[str, Any] | None = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """(状態, 質問) を 1 行ずつ返す。行に questions がなければ、引数の質問セットを使う。"""
    for number, row in _rows(path):
        specs = row.get("questions", questions)
        if not row.get("state") or not specs:
            raise ValueError(f"{path}:{number}: state か questions がありません")
        yield row["state"], specs


def label(decider: Decider, state: str, specs: dict[str, Any]) -> Example:
    """1 つの状態の全質問に、decider の確率を付ける。"""
    answers = decider.decide(state, parse_questions(specs))
    labels: dict[str, Label] = {
        name: a.probability if isinstance(a, NoulAnswer) else a.probabilities
        for name, a in answers.items()
    }
    return Example(state, specs, labels)


def distill(
    decider: Decider,
    states: str | Path,
    out: str | Path,
    questions: dict[str, Any] | None = None,
    log_every: int = 100,
) -> int:
    """状態のファイルを読み、ラベルを付けて out に書く。書いた行数を返す。"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for state, specs in load_states(states, questions):
            f.write(json.dumps(label(decider, state, specs).to_dict(), ensure_ascii=False) + "\n")
            count += 1
            if count % log_every == 0:
                print(f"{count} 行", file=sys.stderr)
    return count


def _check(example: Example) -> None:
    parsed = parse_questions(example.questions)
    if example.labels.keys() != parsed.keys():
        raise ValueError("labels と questions の質問名が合いません")
    for name, q in parsed.items():
        value = example.labels[name]
        if isinstance(q, Choice):
            if not isinstance(value, dict) or set(value) != set(q.options):
                raise ValueError(f"{name}: labels の選択肢が options と合いません")
            if not math.isclose(sum(value.values()), 1.0, abs_tol=1e-3):
                raise ValueError(f"{name}: 選択肢の確率の合計が 1 ではありません")
            probabilities = list(value.values())
        else:
            if isinstance(value, dict):
                raise ValueError(f"{name}: noul のラベルは確率 (0〜1) で書きます")
            probabilities = [value]
        if not all(0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError(f"{name}: 確率が 0〜1 の範囲にありません")


def load_examples(path: str | Path) -> list[Example]:
    """distill の出力と同じ形式の JSONL を読む。形式が合わない行は、行番号を付けてエラーにする。"""
    examples = []
    for number, row in _rows(path):
        try:
            example = Example(row["state"], row["questions"], row["labels"])
            _check(example)
        except (KeyError, ValueError) as e:
            raise ValueError(f"{path}:{number}: {e}") from e
        examples.append(example)
    return examples
