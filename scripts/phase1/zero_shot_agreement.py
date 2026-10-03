"""フェーズ 1: 学習前 (zero-shot) のパック方式が、ペア方式とどれだけ一致するかを測る。

評価セットはフェーズ 0 と同じ ja_eval.yaml。1 件につき noul 3 問 + topic (5 択) を 1 回で推論する。
マーカーの位置 (cls / after) と choice の詰め方 (shared / expanded) の組み合わせを比べる。

    workshop exec gpu -- bash -c \
        'cd /project && uv run python scripts/phase1/zero_shot_agreement.py'
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from local_decision_model.schema import Choice, Noul
from local_decision_model.scoring import DEFAULT_BASE_MODEL, InferenceConfig
from local_decision_model.torch_backend import TorchPackedScorer, TorchScorer

DATA = Path(__file__).parents[1] / "phase0" / "ja_eval.yaml"
NOUL_KINDS = ["平叙文", "疑問文"]
TOPIC_TEMPLATES = {
    "平叙文": "この問い合わせは{option}に関するものだ。",
    "疑問文": "この問い合わせは{option}に関するものですか？",
    "疑問文+選択肢": "この問い合わせは何に関するものですか？ {option}",
}


def questions_for(item, topics, noul_col: int, template: str):
    qs = {f"n{i}": Noul(q[noul_col]) for i, q in enumerate(item["noul"])}
    qs["topic"] = Choice(template, tuple(topics))
    return qs


def pair_logits(scorer, state, questions):
    hyps = {name: q.hypotheses() for name, q in questions.items()}
    z = iter(scorer.score([(state, h) for hs in hyps.values() for h in hs]))
    return {name: np.array([next(z) for _ in hs]) for name, hs in hyps.items()}


def collect(score, items, topics, noul_col, template):
    """(noul のロジット [N], topic のロジット [件数, 選択肢])。"""
    noul, topic = [], []
    for it in items:
        z = score(it["state"], questions_for(it, topics, noul_col, template))
        noul += [z[f"n{i}"][0] for i in range(len(it["noul"]))]
        topic.append(z["topic"])
    return np.array(noul), np.stack(topic)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bench(fn, runs=30, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    topics, items = data["topics"], data["items"]
    noul_gold = np.array([bool(q[2]) for it in items for q in it["noul"]])
    topic_gold = np.array([topics.index(it["topic"]) for it in items])

    pair = TorchScorer(args.model, args.device)
    packed = TorchPackedScorer(args.model, args.device)

    def pair_score(state, questions):
        return pair_logits(pair, state, questions)

    # ペア方式の基準 (noul は文の種類ごと、topic はテンプレートごと)
    pair_noul = {
        k: collect(pair_score, items, topics, c, "{option}")[0] for c, k in enumerate(NOUL_KINDS)
    }
    pair_topic = {
        k: collect(pair_score, items, topics, 0, t)[1] for k, t in TOPIC_TEMPLATES.items()
    }
    print("== ペア方式")
    for k, z in pair_noul.items():
        print(f"  noul  {k}: acc={np.mean((z > 0) == noul_gold):.3f}")
    for k, z in pair_topic.items():
        print(f"  topic {k}: acc={np.mean(z.argmax(1) == topic_gold):.3f}")

    for marker in ("cls", "after"):
        for layout in ("shared", "expanded"):
            packed.config = InferenceConfig(
                arch="packed", marker_position=marker, choice_layout=layout
            )
            print(f"\n== パック方式  marker_position={marker}  choice_layout={layout}")
            for col, kind in enumerate(NOUL_KINDS):
                z, _ = collect(packed.score_questions, items, topics, col, "{option}")
                ref = pair_noul[kind]
                print(
                    f"  noul  {kind}: acc={np.mean((z > 0) == noul_gold):.3f}"
                    f"  ペアと同じ判定={np.mean((z > 0) == (ref > 0)):.3f}"
                    f"  |Δp| 平均={np.mean(np.abs(sigmoid(z) - sigmoid(ref))):.3f}"
                    f"  z の相関={np.corrcoef(z, ref)[0, 1]:.3f}"
                )
            for kind, template in TOPIC_TEMPLATES.items():
                _, z = collect(packed.score_questions, items, topics, 0, template)
                ref = pair_topic[kind]
                print(
                    f"  topic {kind}: acc={np.mean(z.argmax(1) == topic_gold):.3f}"
                    f"  ペアと同じ選択={np.mean(z.argmax(1) == ref.argmax(1)):.3f}"
                )

    # 速さ: 1 件 (noul 3 問 + 5 択 = ペア方式では 8 ペア)
    it = items[0]
    qs = questions_for(it, topics, 0, TOPIC_TEMPLATES["平叙文"])
    print(f"\n== 速さ ({args.device}、1 件 = noul 3 問 + 5 択、p50)")
    ms = bench(lambda: pair_score(it["state"], qs))
    print(f"  ペア方式: {ms:.1f} ms (8 ペアを 1 バッチ、長さ 256)")
    for layout in ("shared", "expanded"):
        packed.config = InferenceConfig(arch="packed", choice_layout=layout)
        n = len(packed_lengths(packed, it["state"], qs))
        ms = bench(lambda: packed.score_questions(it["state"], qs))
        print(f"  パック方式 ({layout}): {ms:.1f} ms ({n} 回の推論)")


def packed_lengths(scorer, state, questions) -> list[int]:
    from local_decision_model.packing import pack

    return [len(p.input_ids) for p in pack(scorer.tokenizer, state, questions, scorer.config)]


if __name__ == "__main__":
    main()
