"""フェーズ 0: 日本語の評価セット (ja_eval.yaml) で zero-shot の精度を測る。

- noul: 平叙文の仮説と、jev の instructions のような疑問文とで、正解率・NLL・ECE を比べる
- topic: choice (1 つ選ぶ) の正解率。仮説文は平叙文と疑問文の 2 通り

    workshop exec gpu -- bash -c 'cd /project && uv run python scripts/phase0/zero_shot_ja.py'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from local_decision_model.calibration import binary_metrics
from local_decision_model.decide import softmax
from local_decision_model.torch_backend import TorchScorer

MODELS = [
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
]
DATA = Path(__file__).with_name("ja_eval.yaml")
TOPIC_TEMPLATES = {
    "平叙文": "この問い合わせは{option}に関するものだ。",
    "疑問文": "この問い合わせは{option}に関するものですか？",
    # jev の choice のように instructions に {option} がなく、選択肢を後ろにつなげる形
    "疑問文+選択肢": "この問い合わせは何に関するものですか？ {option}",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    topics, items = data["topics"], data["items"]
    labels = np.array([float(q[2]) for it in items for q in it["noul"]])
    print(f"noul: {len(labels)} 問 (真 {int(labels.sum())}), topic: {len(items)} 問\n")

    for model in args.models:
        scorer = TorchScorer(model, args.device)
        print(f"== {model}")
        for col, kind in [(0, "平叙文"), (1, "疑問文")]:
            z = scorer.score([(it["state"], q[col]) for it in items for q in it["noul"]])
            m = binary_metrics(z, labels)
            print(
                f"  noul  {kind}: acc={m['accuracy']:.3f}  nll={m['nll']:.3f}  ece={m['ece']:.3f}"
                f"  (平均 P(真)={np.mean(1 / (1 + np.exp(-z))):.2f})"
            )
        for kind, template in TOPIC_TEMPLATES.items():
            hyps = [template.replace("{option}", t) for t in topics]
            z = scorer.score([(it["state"], h) for it in items for h in hyps])
            z = z.reshape(len(items), len(topics))
            gold = np.array([topics.index(it["topic"]) for it in items])
            probs = np.stack([softmax(row) for row in z])
            nll = -np.mean(np.log(probs[np.arange(len(items)), gold] + 1e-12))
            acc = np.mean(z.argmax(1) == gold)
            print(f"  topic {kind}: acc={acc:.3f}  nll={nll:.3f}")
        print()


if __name__ == "__main__":
    main()
