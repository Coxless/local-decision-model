"""質問 (型) と回答の定義。

TypeSafe の System One API にならい、「状態 (自由テキスト) + 名前付きの質問」を入力に、
質問ごとに型付きで確率の付いた回答を返す。文字列は生成しない。

- Noul: 確率付きの真偽値。instructions は「真であるとき成り立つ文」を書く。
- Choice: 選択肢から 1 つを選ぶ。各選択肢を仮説文にして比較する。

質問の書き方は System One API と同じ (``criteria``)。
choice の ``options`` と ``{option}`` は独自の書き方。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MAX_OPTIONS = 255  # System One API の choice の上限


@dataclass(frozen=True)
class Noul:
    """確率付きの真偽値 (System One の Noul 相当)。

    criteria_true (System One の ``criteria.true``) があれば、instructions の代わりに仮説文にする。
    """

    instructions: str
    criteria_true: str | None = None

    def hypotheses(self) -> list[str]:
        return [self.criteria_true or self.instructions]


@dataclass(frozen=True)
class Choice:
    """選択肢から 1 つを選ぶ質問。

    instructions に ``{option}`` があれば置き換え、なければ末尾に選択肢をつなげて仮説文にする。
    descriptions (System One の ``criteria`` の値) は選択肢の説明で、選択肢と同じ順に並べる。
    説明のある選択肢は「選択肢 (説明)」の形で仮説文に入れる。
    """

    instructions: str
    options: tuple[str, ...]
    descriptions: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("Choice には選択肢が 2 つ以上必要です")
        if len(self.options) > MAX_OPTIONS:
            raise ValueError(f"Choice の選択肢は {MAX_OPTIONS} 個までです")
        if len(set(self.options)) != len(self.options):
            raise ValueError("Choice の選択肢が重複しています")
        if self.descriptions and len(self.descriptions) != len(self.options):
            raise ValueError("Choice の説明の数が選択肢の数と合いません")

    def option_texts(self) -> list[str]:
        """仮説文に入れる選択肢の文字列 (選択肢の順)。"""
        if not self.descriptions:
            return list(self.options)
        return [
            f"{o} ({d})" if d else o for o, d in zip(self.options, self.descriptions, strict=True)
        ]

    def reordered(self, options: list[str]) -> Choice:
        """選択肢を並べ替えた Choice (説明も一緒に並べ替える)。"""
        if not self.descriptions:
            return Choice(self.instructions, tuple(options))
        descriptions = dict(zip(self.options, self.descriptions, strict=True))
        return Choice(self.instructions, tuple(options), tuple(descriptions[o] for o in options))

    def hypotheses(self) -> list[str]:
        texts = self.option_texts()
        if "{option}" in self.instructions:
            return [self.instructions.replace("{option}", t) for t in texts]
        if not self.instructions:
            return texts
        return [f"{self.instructions} {t}" for t in texts]


Question = Noul | Choice


@dataclass(frozen=True)
class NoulAnswer:
    probability: float

    @property
    def value(self) -> bool:
        return self.probability >= 0.5

    def to_dict(self) -> dict[str, Any]:
        return {"type": "noul", "noul": self.probability}


@dataclass(frozen=True)
class ChoiceAnswer:
    probabilities: dict[str, float]

    @property
    def value(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def confidence(self) -> float:
        """確率が 1 つの選択肢にどれだけ集中しているか。一様なら 0、1 つに決まっていれば 1。"""
        n = len(self.probabilities)
        p_max = max(self.probabilities.values())
        return min(1.0, max(0.0, (n * p_max - 1) / (n - 1)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.value,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


Answer = NoulAnswer | ChoiceAnswer


def state_text(state: Any) -> str:
    """状態を文字列にする。オブジェクトや配列は JSON 文字列にする。"""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def parse_question(spec: str | dict[str, Any]) -> Question:
    """YAML / JSON の 1 項目から質問を作る。文字列だけなら Noul とみなす。

    type は noul / choice。以前の形式の bool も noul として受け付ける。
    choice の選択肢は ``criteria`` (選択肢 → 説明か null) か ``options`` (選択肢の配列) で書く。
    """
    if isinstance(spec, str):
        return Noul(spec)
    kind = spec.get("type", "noul")
    instructions = spec.get("instructions") or ""
    criteria = spec.get("criteria")
    if kind in ("noul", "bool"):
        criteria_true = (criteria or {}).get("true")
        if not instructions and not criteria_true:
            raise ValueError("noul には instructions か criteria.true が必要です")
        return Noul(instructions, criteria_true)
    if kind == "choice":
        if criteria is not None:
            return Choice(instructions, tuple(criteria), tuple(criteria.values()))
        return Choice(instructions, tuple(spec["options"]))
    raise ValueError(f"未知の質問タイプです: {kind!r} (noul / choice)")


def parse_questions(specs: dict[str, Any]) -> dict[str, Question]:
    if not specs:
        raise ValueError("questions が空です")
    return {name: parse_question(spec) for name, spec in specs.items()}
