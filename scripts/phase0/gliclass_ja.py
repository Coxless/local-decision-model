"""フェーズ 0 (比較): GLiClass の多言語版で ja_eval.yaml を 1 パスで分類する。

gliclass はプロジェクトの依存に入れず、一時的に足して実行する:

    workshop exec gpu -- bash -c \
      'cd /project && uv run --with gliclass python scripts/phase0/gliclass_ja.py'
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

MODELS = [
    "knowledgator/gliclass-multilang-mini",
    "knowledgator/gliclass-multilang-edge",
    "knowledgator/gliclass-x-base",
]
DATA = Path(__file__).with_name("ja_eval.yaml")


def evaluate(model_id: str, device: str, topics: list[str], items: list[dict]) -> None:
    model = GLiClassModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
    single = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="single-label", device=device
    )
    multi = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="multi-label", device=device
    )

    texts = [it["state"] for it in items]
    gold = [it["topic"] for it in items]
    single(texts[:2], topics, threshold=0.0)  # ウォームアップ
    start = time.perf_counter()
    results = single(texts, topics, threshold=0.0)
    ms = (time.perf_counter() - start) * 1000 / len(texts)
    preds = [max(r, key=lambda x: x["score"])["label"] for r in results]
    acc = np.mean([a == b for a, b in zip(preds, gold, strict=True)])
    print(f"  topic: acc={acc:.3f}  ({ms:.1f} ms/件)")

    # noul を「ラベル 1 個の多ラベル分類」とみなしたときの精度 (平叙文を使う)
    hits = []
    for it in items:
        for hyp, _, label in it["noul"]:
            r = multi(it["state"], [hyp], threshold=0.0)[0]
            score = r[0]["score"] if r else 0.0
            hits.append((score >= 0.5) == label)
    print(f"  noul (ラベル 1 個): acc={np.mean(hits):.3f}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    data = yaml.safe_load(DATA.read_text(encoding="utf-8"))

    for model_id in args.models:
        print(f"== {model_id}")
        try:
            evaluate(model_id, args.device, data["topics"], data["items"])
        except Exception as e:  # noqa: BLE001 (比較用なので動かないモデルは飛ばす)
            print(f"  動きません: {type(e).__name__}: {str(e).splitlines()[0]}\n")


if __name__ == "__main__":
    main()
