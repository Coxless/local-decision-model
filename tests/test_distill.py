import json

import numpy as np
import pytest

from local_decision_model.decide import Decider
from local_decision_model.distill import distill, load_examples, load_question_set

QUESTIONS = {
    "refund": "返金？",
    "topic": {"type": "choice", "instructions": "{option}の話", "options": ["配送", "品質"]},
}


class FakeScorer:
    """仮説文 → 判断ロジットの固定表。"""

    def __init__(self, table):
        self.table = table

    def score(self, pairs):
        return np.array([self.table[h] for _, h in pairs])


def decider(temperature=1.0):
    return Decider(FakeScorer({"返金？": 2.0, "配送の話": 1.0, "品質の話": 0.0}), temperature)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def test_distill_writes_labels_for_every_question(tmp_path):
    states = write_jsonl(tmp_path / "states.jsonl", [{"state": "a"}, {"state": "b"}])
    out = tmp_path / "sub" / "train.jsonl"
    assert distill(decider(), states, out, QUESTIONS) == 2

    examples = load_examples(out)
    assert [e.state for e in examples] == ["a", "b"]
    assert examples[0].questions == QUESTIONS
    assert examples[0].labels["refund"] == pytest.approx(1 / (1 + np.exp(-2.0)))
    topic = examples[0].labels["topic"]
    assert list(topic) == ["配送", "品質"]
    assert sum(topic.values()) == pytest.approx(1.0)
    assert topic["配送"] == pytest.approx(np.e / (np.e + 1))


def test_distill_uses_teacher_temperature(tmp_path):
    states = write_jsonl(tmp_path / "states.jsonl", [{"state": "a"}])
    distill(decider(4.0), states, tmp_path / "out.jsonl", QUESTIONS)
    labels = load_examples(tmp_path / "out.jsonl")[0].labels
    assert labels["refund"] == pytest.approx(1 / (1 + np.exp(-0.5)))


def test_row_questions_override_question_set(tmp_path):
    rows = [{"state": "a", "questions": {"only": "返金？"}}, {"state": "b"}]
    states = write_jsonl(tmp_path / "states.jsonl", rows)
    distill(decider(), states, tmp_path / "out.jsonl", QUESTIONS)
    first, second = load_examples(tmp_path / "out.jsonl")
    assert list(first.labels) == ["only"]
    assert list(second.labels) == ["refund", "topic"]


def test_state_without_questions_is_an_error(tmp_path):
    states = write_jsonl(tmp_path / "states.jsonl", [{"state": "a"}])
    with pytest.raises(ValueError, match="states.jsonl:1"):
        distill(decider(), states, tmp_path / "out.jsonl")


def test_load_question_set(tmp_path):
    path = tmp_path / "questions.yaml"
    path.write_text("questions:\n  refund: 返金？\n", encoding="utf-8")
    assert load_question_set(path) == {"refund": "返金？"}
    path.write_text("questions:\n  topic: {type: choice, instructions: x, options: [a]}\n")
    with pytest.raises(ValueError):
        load_question_set(path)


@pytest.mark.parametrize(
    "labels",
    [
        {"refund": 0.9},  # 質問が足りない
        {"refund": 0.9, "topic": {"配送": 1.0}},  # 選択肢が足りない
        {"refund": 0.9, "topic": {"配送": 0.9, "品質": 0.9}},  # 合計が 1 でない
        {"refund": 1.5, "topic": {"配送": 0.5, "品質": 0.5}},  # 範囲外
        {"refund": {"配送": 1.0}, "topic": {"配送": 0.5, "品質": 0.5}},  # noul に分布
    ],
)
def test_load_examples_rejects_bad_labels(tmp_path, labels):
    path = write_jsonl(
        tmp_path / "bad.jsonl", [{"state": "a", "questions": QUESTIONS, "labels": labels}]
    )
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_examples(path)


def test_load_examples_accepts_hard_labels(tmp_path):
    labels = {"refund": 1, "topic": {"配送": 0, "品質": 1}}
    path = write_jsonl(
        tmp_path / "gold.jsonl", [{"state": "a", "questions": QUESTIONS, "labels": labels}]
    )
    assert load_examples(path)[0].labels == labels
