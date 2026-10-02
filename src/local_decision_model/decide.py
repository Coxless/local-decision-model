"""状態 + 質問 → 型付き・確率付きの回答。"""

from __future__ import annotations

import numpy as np

from .schema import Answer, BoolAnswer, Choice, ChoiceAnswer, Noul, Question
from .scoring import Pair, Scorer


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Decider:
    def __init__(self, scorer: Scorer, temperature: float = 1.0):
        self.scorer = scorer
        self.temperature = temperature

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        # 全質問の仮説をまとめて 1 回で推論する
        pairs: list[Pair] = []
        spans: dict[str, slice] = {}
        for name, q in questions.items():
            hyps = q.hypotheses()
            spans[name] = slice(len(pairs), len(pairs) + len(hyps))
            pairs.extend((state, h) for h in hyps)

        z = np.asarray(self.scorer.score(pairs), dtype=np.float64) / self.temperature

        answers: dict[str, Answer] = {}
        for name, q in questions.items():
            zq = z[spans[name]]
            if isinstance(q, Noul):
                answers[name] = BoolAnswer(float(sigmoid(zq[0])))
            elif isinstance(q, Choice):
                probs = softmax(zq)
                answers[name] = ChoiceAnswer(
                    {o: float(p) for o, p in zip(q.options, probs, strict=True)}
                )
        return answers
