"""バックエンド共通の処理: 入力ペアのトークナイズと、分類ロジットから判断ロジットへの変換。

判断ロジット z は「仮説 (質問) が状態から成り立つ」度合いで、P(真) = sigmoid(z / T)。
NLI モデル (entailment / neutral / contradiction) では z = logit[entailment] - logit[contradiction]
とする (zero-shot 分類と同じ考え方)。ファインチューニングもこの z に対して行う。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_BASE_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CONFIG_FILE = "s1.json"

Pair = tuple[str, str]  # (状態, 仮説)


@dataclass
class InferenceConfig:
    """モデルディレクトリに保存する推論設定。"""

    arch: str = "pair"  # "pair": (状態, 仮説) ごとに推論 / "packed": 1 本の系列に詰めて推論
    max_length: int = 256  # ペア方式の系列長
    temperature_noul: float = 1.0
    temperature_choice: float = 1.0
    # 以下はパック方式 (packing.py)
    lengths: list[int] = field(default_factory=lambda: [128, 256, 512])
    max_markers: int = 32
    max_state_tokens: int = 384  # [CLS] と [SEP] を含む
    marker_position: str = "cls"  # マーカーの位置番号。"cls": 0 / "after": 質問の先頭
    choice_layout: str = "shared"  # "shared": 指示を共有 / "expanded": 選択肢ごとの仮説文
    state_sees_questions: bool = False  # 真: 状態も質問を見る (比較実験用。質問同士が干渉する)

    @classmethod
    def load(cls, model_dir: str | Path) -> InferenceConfig:
        path = Path(model_dir) / CONFIG_FILE
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        if "temperature" in data:  # 以前の形式: 温度は 1 つ
            temperature = data.pop("temperature")
            data.setdefault("temperature_noul", temperature)
            data.setdefault("temperature_choice", temperature)
        if "temperature_bool" in data:  # 以前の形式: noul の温度が temperature_bool
            data.setdefault("temperature_noul", data.pop("temperature_bool"))
        return cls(**data)

    def save(self, model_dir: str | Path) -> None:
        path = Path(model_dir) / CONFIG_FILE
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


class Scorer(Protocol):
    """(状態, 仮説) ペアごとの判断ロジットを返すもの。"""

    def score(self, pairs: list[Pair]) -> np.ndarray: ...


class QuestionScorer(Protocol):
    """状態と質問から、質問名 → 判断ロジット (choice は選択肢の順) を直接返すもの (パック方式)。"""

    def score_questions(self, state: str, questions: dict) -> dict[str, np.ndarray]: ...


def decision_logits(logits, id2label: dict[int, str]):
    """分類ロジット (N, num_labels) を判断ロジット (N,) にする。numpy でも torch でもよい。"""
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

    長すぎる場合は状態の側だけを切り詰め、仮説は残す。
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
