"""状態 + 質問 → 型付き・確率付きの回答。"""

from __future__ import annotations

import numpy as np

from .schema import Answer, Choice, ChoiceAnswer, Noul, NoulAnswer, Question
from .scoring import Pair, QuestionScorer, Scorer


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Decider:
    """scorer はペア方式 (Scorer) かパック方式 (QuestionScorer) のどちらか。"""

    def __init__(
        self,
        scorer: Scorer | QuestionScorer,
        temperature: float = 1.0,
        temperature_choice: float | None = None,
    ):
        self.scorer = scorer
        self.temperature = temperature
        self.temperature_choice = temperature if temperature_choice is None else temperature_choice

    def _score_pairs(self, state: str, questions: dict[str, Question]) -> dict[str, np.ndarray]:
        # 全質問の仮説をまとめて 1 回で推論する
        pairs: list[Pair] = []
        spans: dict[str, slice] = {}
        for name, q in questions.items():
            hyps = q.hypotheses()
            spans[name] = slice(len(pairs), len(pairs) + len(hyps))
            pairs.extend((state, h) for h in hyps)
        z = np.asarray(self.scorer.score(pairs))
        return {name: z[span] for name, span in spans.items()}

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        if hasattr(self.scorer, "score_questions"):
            z = self.scorer.score_questions(state, questions)
        else:
            z = self._score_pairs(state, questions)

        answers: dict[str, Answer] = {}
        for name, q in questions.items():
            zq = np.asarray(z[name], dtype=np.float64)
            if isinstance(q, Noul):
                answers[name] = NoulAnswer(float(sigmoid(zq[0] / self.temperature)))
            elif isinstance(q, Choice):
                probs = softmax(zq / self.temperature_choice)
                answers[name] = ChoiceAnswer(
                    {o: float(p) for o, p in zip(q.options, probs, strict=True)}
                )
        return answers
