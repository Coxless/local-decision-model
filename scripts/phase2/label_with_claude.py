"""フェーズ 2: 大きな LLM の教師。状態と質問セットに、Claude Code でラベルの確率を付ける。

出力は distill と同じ形式の JSONL (1 行 = 1 つの状態 + 複数の質問 + labels) なので、
ペア方式の教師のラベルと行ごとに比べたり、混ぜたりできる。

Claude には確率を 0〜100 の整数で答えさせる (noul は 1 つ、choice は選択肢ごと)。
choice は合計が 1 になるように割り直して保存する。

途中で止めても、同じコマンドでもう一度実行すれば、まだラベルのない状態だけを足す。
Claude Code (`claude` コマンド) にログインしたホストで、ワークショップの外から実行する
(PyYAML が要る):

    python3 scripts/phase2/label_with_claude.py data/states.jsonl \
        --questions data/questions.yaml --out data/train-claude.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

SYSTEM = (
    "あなたは、顧客サポートに届いた問い合わせ文を読んで質問に答える、注意深い判定者です。"
    "機械学習の教師ラベルに使うので、文面に書かれていることだけから、確率で答えてください。"
)

PROMPT = """問い合わせ文ごとに、下の質問すべてに答えてください。

## noul の質問 (その文が成り立つ確率を 0〜100 の整数で答える)
{nouls}

## choice の質問 (選択肢ごとの確率を、選択肢の順に並べた整数の配列で答える。合計は 100)
{choices}

## 答え方
- 問い合わせ文に書かれていることだけから判断する。書かれていないことを補って考えない
- はっきり成り立つなら 95〜100、はっきり成り立たないなら 0〜5 にする
- 読む人によって判断が分かれるときは、成り立つと判断する人の割合を中間の値で表す
- 出力は JSON の配列だけにする。問い合わせ文 1 件につき 1 つのオブジェクトを書き、
  どの問い合わせ文への答えかを "i" に番号で書く。{n} 件すべてに答える
  例: {example}

## 問い合わせ文 ({n} 件)
{states}"""


def options_of(spec: str | dict[str, Any]) -> list[str] | None:
    """choice なら選択肢、noul なら None。"""
    if isinstance(spec, dict) and spec.get("type") == "choice":
        return list(spec["options"])
    return None


def instructions_of(spec: str | dict[str, Any]) -> str:
    return spec if isinstance(spec, str) else spec["instructions"]


def build_prompt(states: list[str], questions: dict[str, Any]) -> str:
    nouls, choices, example = [], [], {}
    for name, spec in questions.items():
        options = options_of(spec)
        if options is None:
            nouls.append(f"- {name}: {instructions_of(spec)}")
            example.setdefault(name, 90)
        else:
            choices.append(f"- {name}: {instructions_of(spec)} 選択肢: {options}")
            example.setdefault(name, [100 // len(options)] * len(options))
    shown = {"i": 1} | dict(list(example.items())[:2] + list(example.items())[-1:])
    numbered = [{"i": i, "text": s} for i, s in enumerate(states, 1)]
    return PROMPT.format(
        nouls="\n".join(nouls),
        choices="\n".join(choices),
        example=json.dumps(shown, ensure_ascii=False).replace("}", ", ...}"),
        n=len(states),
        states=json.dumps(numbered, ensure_ascii=False, indent=1),
    )


def to_labels(answer: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
    """Claude の答え (0〜100) を labels (0〜1) にする。形が合わなければ例外。"""
    labels: dict[str, Any] = {}
    for name, spec in questions.items():
        value = answer[name]
        options = options_of(spec)
        if options is None:
            if not isinstance(value, int | float) or not 0 <= value <= 100:
                raise ValueError(f"{name}: 0〜100 の数ではありません: {value!r}")
            labels[name] = value / 100
        else:
            ok = isinstance(value, list) and len(value) == len(options)
            if not ok or any(not isinstance(v, int | float) or v < 0 for v in value):
                raise ValueError(f"{name}: 選択肢ごとの確率の配列ではありません: {value!r}")
            total = sum(value)
            if total <= 0:
                raise ValueError(f"{name}: 確率がすべて 0 です")
            labels[name] = {o: v / total for o, v in zip(options, value, strict=True)}
    return labels


def label_batch(
    states: list[str], questions: dict[str, Any], model: str, workdir: str
) -> dict[str, dict[str, Any]]:
    """Claude Code を 1 回呼び、状態 → labels を返す。

    答えが抜けた状態や、形が合わない答えは返さない (もう一度実行したときに付け直す)。
    """
    result = subprocess.run(
        ["claude", "-p", "--model", model, "--system-prompt", SYSTEM]
        + ["--tools", "", "--strict-mcp-config", "--disable-slash-commands"]
        + ["--no-session-persistence", "--output-format", "text"],
        input=build_prompt(states, questions),
        capture_output=True,
        text=True,
        cwd=workdir,  # プロジェクトの CLAUDE.md を読み込ませない
        check=True,
    )
    out = result.stdout
    labels = {}
    for answer in json.loads(out[out.index("[") : out.rindex("]") + 1]):
        try:
            labels[states[answer["i"] - 1]] = to_labels(answer, questions)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"形が合わない答えを飛ばします: {e!r}", file=sys.stderr)
    return labels


def read_states(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["state"] for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("states", help='JSONL。1 行 = {"state": "..."}')
    p.add_argument("--questions", default="data/questions.yaml")
    p.add_argument("--out", default="data/train-claude.jsonl")
    p.add_argument("--batch", type=int, default=10, help="1 回の呼び出しで答えさせる状態の数")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--limit", type=int, help="先頭からこの件数だけ処理する (試し用)")
    args = p.parse_args()

    questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(read_states(out)) if out.exists() else set()
    todo = [s for s in read_states(Path(args.states))[: args.limit] if s not in done]
    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]

    count, failed = len(done), 0
    with (
        tempfile.TemporaryDirectory() as workdir,
        ThreadPoolExecutor(args.workers) as pool,
        open(out, "a", encoding="utf-8") as f,
    ):
        futures = {pool.submit(label_batch, b, questions, args.model, workdir): b for b in batches}
        for future in as_completed(futures):
            try:
                labels = future.result()
            except (subprocess.CalledProcessError, ValueError) as e:
                failed += 1
                called = isinstance(e, subprocess.CalledProcessError)
                # 利用上限などのエラーは標準出力に出る
                detail = (e.stderr.strip() or e.stdout.strip()[-200:]) if called else e
                print(f"失敗した呼び出しを飛ばします: {detail!r}", file=sys.stderr)
                continue
            for state, row in labels.items():
                example = {"state": state, "questions": questions, "labels": row}
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
            f.flush()
            count += len(labels)
            print(f"{count} / {len(done) + len(todo)} 件", file=sys.stderr)

    print(f"{count} 件: {out}")
    if count < len(done) + len(todo):
        print(f"失敗した呼び出し: {failed} 回。もう一度実行すると、残りだけを足します。")


if __name__ == "__main__":
    main()
