"""質問 (型) と回答の定義。

TypeSafe の System One API にならい、「状態 (自由テキスト) + 名前付きの質問」を入力に、
質問ごとに型付きで確率の付いた回答を返す。文字列は生成しない。

- Noul: 確率付きの真偽値。instructions は「真であるとき成り立つ文」を書く。
- Choice: 選択肢から 1 つを選ぶ。各選択肢を仮説文にして比較する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Noul:
    """確率付きの真偽値 (System One の Noul 相当)。"""

    instructions: str

    def hypotheses(self) -> list[str]:
        return [self.instructions]


@dataclass(frozen=True)
class Choice:
    """選択肢から 1 つを選ぶ質問。

    instructions に ``{option}`` があれば置き換え、なければ末尾に選択肢をつなげて仮説文にする。
    """

    instructions: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("Choice には選択肢が 2 つ以上必要です")
        if len(set(self.options)) != len(self.options):
            raise ValueError("Choice の選択肢が重複しています")

    def hypotheses(self) -> list[str]:
        if "{option}" in self.instructions:
            return [self.instructions.replace("{option}", o) for o in self.options]
        return [f"{self.instructions} {o}" for o in self.options]


Question = Noul | Choice


@dataclass(frozen=True)
class NoulAnswer:
    probability: float

    @property
    def value(self) -> bool:
        return self.probability >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {"type": "noul", "value": self.value, "probability": self.probability}


@dataclass(frozen=True)
class ChoiceAnswer:
    probabilities: dict[str, float]

    @property
    def value(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "choice", "value": self.value, "probabilities": self.probabilities}


Answer = NoulAnswer | ChoiceAnswer


def parse_question(spec: str | dict[str, Any]) -> Question:
    """YAML / JSON の 1 項目から質問を作る。文字列だけなら Noul とみなす。

    type は noul / choice。以前の形式の bool も noul として受け付ける。
    """
    if isinstance(spec, str):
        return Noul(spec)
    kind = spec.get("type", "noul")
    if kind in ("noul", "bool"):
        return Noul(spec["instructions"])
    if kind == "choice":
        return Choice(spec["instructions"], tuple(spec["options"]))
    raise ValueError(f"未知の質問タイプです: {kind!r} (noul / choice)")


def parse_questions(specs: dict[str, Any]) -> dict[str, Question]:
    if not specs:
        raise ValueError("questions が空です")
    return {name: parse_question(spec) for name, spec in specs.items()}
