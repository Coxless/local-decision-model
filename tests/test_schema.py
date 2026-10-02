import pytest

from local_s1.schema import BoolAnswer, Choice, ChoiceAnswer, Noul, parse_questions


def test_parse_questions():
    qs = parse_questions(
        {
            "a": "It is raining.",
            "b": {"type": "bool", "instructions": "It is sunny."},
            "c": {"type": "choice", "instructions": "About {option}.", "options": ["x", "y"]},
        }
    )
    assert qs["a"] == Noul("It is raining.")
    assert qs["b"] == Noul("It is sunny.")
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
    assert BoolAnswer(0.7).to_dict() == {"type": "bool", "value": True, "probability": 0.7}
    assert ChoiceAnswer({"x": 0.2, "y": 0.8}).value == "y"
