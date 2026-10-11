"""パック方式の IR のテスト。torch と OpenVINO の CPU で、同じ入力の判断ロジットが一致すること。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ov = pytest.importorskip("openvino")

from test_packed_model import (  # noqa: E402, F401
    QUESTIONS,
    STATE,
    FakeTokenizer,
    base,
    config,
    score,
)

from local_decision_model.export import convert_packed  # noqa: E402
from local_decision_model.openvino_backend import IR_FILE, OpenVINOPackedScorer  # noqa: E402
from local_decision_model.packed_model import PackedModel  # noqa: E402
from local_decision_model.packing import pack  # noqa: E402
from local_decision_model.schema import Noul  # noqa: E402


@pytest.fixture(scope="module")
def model(base):  # noqa: F811
    model = PackedModel(base).eval()
    with torch.no_grad():
        # 役割埋め込みが IR に入っていることも確かめるので、0 のままにしない
        model.role_embeddings.weight.normal_()
    return model


@pytest.fixture(scope="module")
def model_dir(model, tmp_path_factory):
    """一番短い系列長の例でトレースした IR。ほかの系列長は読み込み側の reshape で作る。"""
    out = tmp_path_factory.mktemp("packed-ov")
    (example,) = pack(FakeTokenizer(), "s", {"q": Noul("h")}, config())
    assert len(example.input_ids) == 32
    ov.save_model(convert_packed(model, example.arrays()), out / IR_FILE, compress_to_fp16=False)
    return out


def scorer(model_dir, cfg):
    return OpenVINOPackedScorer(model_dir, "CPU", cfg, cache_dir=None, tokenizer=FakeTokenizer())


FEW = {name: QUESTIONS[name] for name in ("a", "b")}


@pytest.mark.parametrize(
    ("state", "questions", "cfg", "length"),
    [
        ("s", {"a": QUESTIONS["a"]}, config(), 32),
        (STATE, FEW, config(), 64),
        (STATE, QUESTIONS, config(), 128),
        (STATE, QUESTIONS, config(marker_position="after", choice_layout="expanded"), 128),
    ],
)
def test_matches_torch(model, model_dir, state, questions, cfg, length):
    expected, packs = score(model, state, questions, cfg)
    assert [len(p.input_ids) for p in packs] == [length]
    z = scorer(model_dir, cfg).score_questions(state, questions)
    for name in questions:
        np.testing.assert_allclose(z[name], expected[name], atol=1e-4)


def test_split_packs_match_torch(model, model_dir):
    cfg = config(max_markers=2)
    expected, packs = score(model, STATE, QUESTIONS, cfg)
    assert len(packs) > 1
    z = scorer(model_dir, cfg).score_questions(STATE, QUESTIONS)
    for name in QUESTIONS:
        np.testing.assert_allclose(z[name], expected[name], atol=1e-4)


def test_compile_all(model_dir):
    s = scorer(model_dir, config())
    s.compile_all()
    assert sorted(s.requests) == [32, 64, 128]
