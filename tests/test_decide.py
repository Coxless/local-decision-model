import numpy as np
import pytest

from local_decision_model.decide import Decider
from local_decision_model.schema import Choice, Noul
from local_decision_model.scoring import CONFIG_FILE, InferenceConfig, decision_logits


class FakeScorer:
    """仮説文 → 判断ロジットの固定表。"""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def score(self, pairs):
        self.calls += 1
        return np.array([self.table[h] for _, h in pairs])


def test_decide_noul_and_choice_in_one_call():
    scorer = FakeScorer({"positive": 2.0, "about a": 1.0, "about b": 1.0})
    answers = Decider(scorer).decide(
        "state", {"p": Noul("positive"), "c": Choice("about {option}", ("a", "b"))}
    )
    assert scorer.calls == 1
    assert answers["p"].probability == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert answers["c"].probabilities == pytest.approx({"a": 0.5, "b": 0.5})


def test_temperature_softens():
    scorer = FakeScorer({"q": 4.0})
    sharp = Decider(scorer, 1.0).decide("s", {"q": Noul("q")})["q"].probability
    soft = Decider(scorer, 4.0).decide("s", {"q": Noul("q")})["q"].probability
    assert 0.5 < soft < sharp


def test_decision_logits_nli():
    logits = np.array([[3.0, 0.0, 1.0]])
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert decision_logits(logits, id2label) == pytest.approx([2.0])


def test_decision_logits_binary():
    assert decision_logits(np.array([[1.0, 4.0]]), {0: "NO", 1: "YES"}) == pytest.approx([3.0])


def test_config_reads_old_temperature_keys(tmp_path):
    (tmp_path / CONFIG_FILE).write_text('{"temperature_bool": 2.0, "temperature_choice": 3.0}')
    config = InferenceConfig.load(tmp_path)
    assert (config.temperature_noul, config.temperature_choice) == (2.0, 3.0)

    (tmp_path / CONFIG_FILE).write_text('{"temperature": 1.5}')
    config = InferenceConfig.load(tmp_path)
    assert (config.temperature_noul, config.temperature_choice) == (1.5, 1.5)
