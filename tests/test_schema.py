import pytest

from local_decision_model.schema import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    parse_questions,
    state_text,
)


def test_parse_questions():
    qs = parse_questions(
        {
            "a": "It is raining.",
            "b": {"type": "noul", "instructions": "It is sunny."},
            "old": {"type": "bool", "instructions": "It is sunny."},  # 以前の形式
            "c": {"type": "choice", "instructions": "About {option}.", "options": ["x", "y"]},
        }
    )
    assert qs["a"] == Noul("It is raining.")
    assert qs["b"] == qs["old"] == Noul("It is sunny.")
    assert qs["c"] == Choice("About {option}.", ("x", "y"))


def test_choice_hypotheses():
    assert Choice("About {option}.", ("x", "y")).hypotheses() == ["About x.", "About y."]
    assert Choice("Topic:", ("x", "y")).hypotheses() == ["Topic: x", "Topic: y"]


@pytest.mark.parametrize("options", [("x",), ("x", "x")])
def test_choice_rejects_bad_options(options):
    with pytest.raises(ValueError):
        Choice("q", options)


def test_unknown_type():
    with pytest.raises(ValueError):
        parse_questions({"a": {"type": "int", "instructions": "q"}})


def test_answers():
    assert NoulAnswer(0.7).to_dict() == {"type": "noul", "noul": 0.7}
    answer = ChoiceAnswer({"x": 0.2, "y": 0.8})
    assert answer.value == "y"
    assert answer.to_dict() == {
        "type": "choice",
        "choice": "y",
        "probabilities": {"x": 0.2, "y": 0.8},
        "confidence": pytest.approx(0.6),
    }


def test_confidence():
    # (n·p_max − 1) / (n − 1)。一様なら 0、1 つに決まっていれば 1
    assert ChoiceAnswer({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}).confidence == 0.0
    assert ChoiceAnswer({"a": 1.0, "b": 0.0, "c": 0.0}).confidence == 1.0
    assert ChoiceAnswer({"a": 0.88, "b": 0.12, "c": 0.0}).confidence == pytest.approx(0.82)


def test_criteria():
    qs = parse_questions(
        {
            "n": {
                "type": "noul",
                "instructions": "Does this convey urgency?",
                "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
            },
            "only_criteria": {"type": "noul", "criteria": {"true": "It is urgent."}},
            "c": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments, invoicing, refunds", "sales": None},
            },
        }
    )
    assert qs["n"].hypotheses() == ["Explicitly time-sensitive"]
    assert qs["only_criteria"].hypotheses() == ["It is urgent."]
    assert qs["c"].options == ("billing", "sales")
    assert qs["c"].hypotheses() == [
        "Which team should handle this? billing (Payments, invoicing, refunds)",
        "Which team should handle this? sales",
    ]
    assert qs["c"].reordered(["sales", "billing"]).option_texts() == [
        "sales",
        "billing (Payments, invoicing, refunds)",
    ]


def test_noul_needs_instructions_or_criteria():
    with pytest.raises(ValueError):
        parse_questions({"n": {"type": "noul", "criteria": {"false": "No urgency"}}})


def test_state_text():
    assert state_text("そのまま") == "そのまま"
    assert state_text({"件名": "二重請求", "n": [1, 2]}) == '{"件名": "二重請求", "n": [1, 2]}'
