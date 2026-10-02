"""バックエンド共通の処理: 入力ペアのトークナイズと、分類ロジットから判断ロジットへの変換。

判断ロジット z は「仮説 (質問) が状態から成り立つ」度合いで、P(真) = sigmoid(z / T)。
NLI モデル (entailment / neutral / contradiction) では z = logit[entailment] - logit[contradiction]
とする (zero-shot 分類と同じ考え方)。ファインチューニングもこの z に対して行う。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_BASE_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
CONFIG_FILE = "s1.json"

Pair = tuple[str, str]  # (状態, 仮説)


@dataclass
class S1Config:
    """モデルディレクトリに保存する推論設定。"""

    max_length: int = 256
    temperature: float = 1.0

    @classmethod
    def load(cls, model_dir: str | Path) -> S1Config:
        path = Path(model_dir) / CONFIG_FILE
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))

    def save(self, model_dir: str | Path) -> None:
        path = Path(model_dir) / CONFIG_FILE
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


class Scorer(Protocol):
    """(状態, 仮説) ペアごとの判断ロジットを返すもの。"""

    def score(self, pairs: list[Pair]) -> np.ndarray: ...


def decision_logits(logits: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    """分類ロジット (N, num_labels) を判断ロジット (N,) にする。"""
    labels = {v.lower(): int(k) for k, v in id2label.items()}
    if logits.shape[-1] == 1:
        return logits[:, 0]
    if "entailment" in labels and "contradiction" in labels:
        return logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]
    if logits.shape[-1] == 2:
        return logits[:, 1] - logits[:, 0]
    raise ValueError(f"判断ロジットに変換できないラベル構成です: {id2label}")


def tokenize(tokenizer, pairs: list[Pair], max_length: int, tensors: str = "np"):
    """静的な形状 (max_length) にパディングする。NPU は動的形状が苦手なため常に固定長にする。

    長すぎる場合は状態の側だけを切り詰め、質問文は残す。
    """
    states = [s for s, _ in pairs]
    hypotheses = [h for _, h in pairs]
    return tokenizer(
        states,
        hypotheses,
        padding="max_length",
        truncation="only_first",
        max_length=max_length,
        return_tensors=tensors,
    )
