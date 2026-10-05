import numpy as np
import pytest

from local_decision_model.decide import Decider
from local_decision_model.distill import Example
from local_decision_model.evaluate import Scores, calibrate, metrics, score

QUESTIONS = {
    "refund": "返金？",
    "topic": {"type": "choice", "instructions": "{option}の話", "options": ["配送", "品質"]},
    "size": {"type": "choice", "instructions": "{option}サイズ", "options": ["小", "中", "大"]},
}


class FakeScorer:
    """(状態, 仮説文) → 判断ロジットの固定表。"""

    def __init__(self, table):
        self.table = table

    def score(self, pairs):
        return np.array([self.table[state][h] for state, h in pairs])


def table(refund, topic, size):
    hypotheses = ["返金？", "配送の話", "品質の話", "小サイズ", "中サイズ", "大サイズ"]
    return dict(zip(hypotheses, [refund, *topic, *size], strict=True))


def labels(refund, topic, size):
    return {
        "refund": refund,
        "topic": dict(zip(["配送", "品質"], topic, strict=True)),
        "size": dict(zip(["小", "中", "大"], size, strict=True)),
    }


TABLE = {
    "a": table(2.0, [1.0, 0.0], [0.0, 3.0, 1.0]),
    "b": table(1.0, [0.0, 2.0], [0.0, 0.0, 1.0]),
}
EXAMPLES = [
    Example("a", QUESTIONS, labels(0.9, [0.8, 0.2], [0.1, 0.8, 0.1])),
    Example("b", QUESTIONS, labels(0.1, [0.7, 0.3], [0.0, 0.1, 0.9])),
]


def test_score_collects_logits_by_type():
    scores = score(Decider(FakeScorer(TABLE)), EXAMPLES)
    assert scores.noul_names == ["refund", "refund"]
    np.testing.assert_allclose(scores.noul_z, [2.0, 1.0])
    np.testing.assert_allclose(scores.noul_y, [0.9, 0.1])
    assert scores.choice_names == ["topic", "size", "topic", "size"]
    # 2 択の行は、3 列目が埋め草になる
    np.testing.assert_allclose(scores.choice_z[0], [1.0, 0.0, -np.inf])
    np.testing.assert_allclose(scores.choice_y[0], [0.8, 0.2, 0.0])
    np.testing.assert_allclose(scores.choice_z[1], [0.0, 3.0, 1.0])


def test_scores_save_and_load(tmp_path):
    scores = score(Decider(FakeScorer(TABLE)), EXAMPLES)
    scores.save(tmp_path / "scores.npz")
    loaded = Scores.load(tmp_path / "scores.npz")
    assert loaded.noul_names == scores.noul_names
    assert loaded.choice_names == scores.choice_names
    np.testing.assert_array_equal(loaded.choice_z, scores.choice_z)
    np.testing.assert_array_equal(loaded.noul_y, scores.noul_y)


def test_metrics_and_calibrate():
    scores = score(Decider(FakeScorer(TABLE)), EXAMPLES)
    report = metrics(scores)
    assert report["noul"]["accuracy"] == pytest.approx(0.5)
    assert report["noul"]["count"] == 2
    assert report["choice"]["accuracy"] == pytest.approx(0.75)
    assert report["accuracy_by_question"] == {"refund": 0.5, "topic": 0.5, "size": 1.0}


def test_calibrate_fits_rounded_labels():
    """温度は、教師の確率のあいまいさではなく、判定が当たる割合に合わせる。"""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=5000)
    hard = rng.random(z.size) < 1 / (1 + np.exp(-z / 2.0))
    soft = np.where(hard, 0.7, 0.3)  # 教師は、当たっていても控えめな確率を付ける
    empty = np.zeros((0, 0))
    scores = Scores(["q"] * z.size, z, soft, [], empty, empty)
    t_noul, t_choice = calibrate(scores)
    assert t_noul == pytest.approx(2.0, rel=0.15)
    assert t_choice == 1.0
