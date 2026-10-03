import numpy as np
import pytest

from local_decision_model.packing import pack, unpack
from local_decision_model.schema import Choice, Noul
from local_decision_model.scoring import S1Config

CLS, SEP, PAD = 1, 2, 0


class FakeTokenizer:
    """1 文字 = 1 トークン (ID は文字コード)。"""

    cls_token_id, sep_token_id, pad_token_id = CLS, SEP, PAD

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def ids(text):
    return [ord(c) for c in text]


def config(**kwargs):
    defaults = dict(arch="packed", lengths=[16, 32, 64], max_markers=4, max_state_tokens=10)
    return S1Config(**{**defaults, **kwargs})


def test_layout_noul_and_choice():
    questions = {"n": Noul("xy"), "c": Choice("p{option}s", ("a", "b"))}
    (p,) = pack(FakeTokenizer(), "st", questions, config())

    # [CLS] s t [SEP] | [Q] x y [SEP] | p | [O] a s [SEP] | [O] b s [SEP]
    tokens = [CLS, *ids("st"), SEP, CLS, *ids("xy"), SEP, *ids("p"), CLS, *ids("as"), SEP]
    tokens += [CLS, *ids("bs"), SEP]
    assert len(p.input_ids) == 32
    assert p.input_ids[: len(tokens)].tolist() == tokens
    assert p.input_ids[len(tokens) :].tolist() == [PAD] * (32 - len(tokens))
    assert p.question_ids[: len(tokens)].tolist() == [0] * 4 + [1] * 4 + [2] * 9
    assert p.question_ids[len(tokens) :].tolist() == [-1] * (32 - len(tokens))
    assert p.option_ids[: len(tokens)].tolist() == [0] * 9 + [1] * 4 + [2] * 4
    # マーカーは位置 0、質問は状態のすぐ後ろ (4) から、選択肢は指示の後ろ (5) から
    positions = [0, 1, 2, 3, 0, 4, 5, 6, 4, 0, 5, 6, 7, 0, 5, 6, 7]
    assert p.positions[: len(tokens)].tolist() == positions
    assert p.markers == [("n", 0), ("c", 0), ("c", 1)]
    assert p.marker_index.tolist() == [4, 9, 13, 0]


def test_marker_position_after():
    questions = {"n": Noul("xy"), "c": Choice("p{option}", ("a", "b"))}
    (p,) = pack(FakeTokenizer(), "st", questions, config(marker_position="after"))
    # マーカーも質問の先頭として番号を振る
    assert p.positions[:15].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 5, 6, 7]


def test_choice_without_template_and_expanded():
    tok = FakeTokenizer()
    (p,) = pack(tok, "s", {"c": Choice("q", ("a", "b"))}, config())
    assert p.input_ids[:10].tolist() == [CLS, *ids("s"), SEP, *ids("q"), CLS, 97, SEP, CLS, 98, SEP]

    (p,) = pack(tok, "s", {"c": Choice("{option}!", ("a", "b"))}, config(choice_layout="expanded"))
    # 選択肢ごとに独立した質問になる
    assert p.input_ids[:11].tolist() == [CLS, 115, SEP, CLS, *ids("a!"), SEP, CLS, *ids("b!"), SEP]
    assert p.question_ids[:11].tolist() == [0] * 3 + [1] * 4 + [2] * 4
    assert p.option_ids.tolist() == [0] * 16
    assert p.markers == [("c", 0), ("c", 1)]


def test_state_truncation_does_not_depend_on_questions():
    tok = FakeTokenizer()
    state = "abcdefghijklmnop"
    (one,) = pack(tok, state, {"a": Noul("x")}, config())
    (two,) = pack(tok, state, {"a": Noul("x"), "b": Noul("y" * 30)}, config())
    expected = [CLS, *ids(state[:8]), SEP]
    assert one.input_ids[:10].tolist() == expected
    assert two.input_ids[:10].tolist() == expected


def test_shortest_length_is_chosen():
    tok = FakeTokenizer()
    assert len(pack(tok, "s", {"a": Noul("x")}, config())[0].input_ids) == 16
    assert len(pack(tok, "s", {"a": Noul("x" * 20)}, config())[0].input_ids) == 32
    assert len(pack(tok, "s", {"a": Noul("x" * 40)}, config())[0].input_ids) == 64


def test_split_by_markers_and_tokens():
    tok = FakeTokenizer()
    questions = {"c": Choice("q", tuple("abcdef")), "n": Noul("x")}
    packs = pack(tok, "s", questions, config())
    # マーカー 4 個まで: 選択肢 4 個 / 残り 2 個 + noul。指示はパックごとに繰り返す
    assert [p.markers for p in packs] == [
        [("c", 0), ("c", 1), ("c", 2), ("c", 3)],
        [("c", 4), ("c", 5), ("n", 0)],
    ]
    assert all(p.input_ids[3] == ord("q") for p in packs)

    # トークン数で分かれる場合 (状態 3 + 質問 22 × 3 > 64)
    questions = {k: Noul(k * 20) for k in "abc"}
    packs = pack(tok, "s", questions, config())
    assert [p.markers for p in packs] == [[("a", 0), ("b", 0)], [("c", 0)]]


def test_too_long_question():
    with pytest.raises(ValueError, match="'a'"):
        pack(FakeTokenizer(), "s", {"a": Noul("x" * 70)}, config())


def test_unpack():
    questions = {"c": Choice("q", tuple("abcdef")), "n": Noul("x")}
    packs = pack(FakeTokenizer(), "s", questions, config())
    outputs = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 99.0])]
    z = unpack(packs, outputs, questions)
    assert z["c"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert z["n"].tolist() == [7.0]
