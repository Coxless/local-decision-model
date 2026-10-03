"""パック方式のモデルのテスト。小さなランダム初期化の DeBERTa を CPU で動かす。"""

import numpy as np
import pytest

from local_decision_model.packing import pack, unpack
from local_decision_model.schema import Choice, Noul
from local_decision_model.scoring import S1Config

torch = pytest.importorskip("torch")

from transformers import DebertaV2Config, DebertaV2ForSequenceClassification  # noqa: E402

from local_decision_model.packed_model import PackedModel  # noqa: E402

CLS, SEP, PAD = 1, 2, 0
VOCAB = 200


class FakeTokenizer:
    cls_token_id, sep_token_id, pad_token_id = CLS, SEP, PAD

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + ord(c) % (VOCAB - 3) for c in text]}


@pytest.fixture(scope="module")
def base():
    torch.manual_seed(0)
    config = DebertaV2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=512,
        relative_attention=True,
        position_buckets=16,  # 小さくして、対数バケットに入る距離もテストで通す
        pos_att_type=["p2c", "c2p"],
        share_att_key=True,
        norm_rel_ebd="layer_norm",
        position_biased_input=False,
        type_vocab_size=0,
        # 既定 (0.02) だと出力がほぼ定数になり、一致のテストが意味を持たない
        initializer_range=0.3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )
    return DebertaV2ForSequenceClassification(config).eval()


@pytest.fixture(scope="module")
def model(base):
    return PackedModel(base).eval()


def config(**kwargs):
    defaults = dict(arch="packed", lengths=[32, 64, 128], max_markers=8, max_state_tokens=24)
    return S1Config(**{**defaults, **kwargs})


def score(model, state, questions, cfg):
    packs = pack(FakeTokenizer(), state, questions, cfg)
    with torch.inference_mode():
        outputs = [
            model(**{k: torch.from_numpy(v) for k, v in p.arrays().items()})[0].numpy()
            for p in packs
        ]
    return unpack(packs, outputs, questions), packs


STATE = "the quick brown fox jumps"
QUESTIONS = {
    "a": Noul("is it quick"),
    "c": Choice("it is {option}.", ("red", "brown", "blue")),
    "b": Noul("is it a dog"),
    "d": Choice("pick", ("x", "yy")),
}
VARIANTS = [
    dict(marker_position=m, choice_layout=c)
    for m in ("cls", "after")
    for c in ("shared", "expanded")
]


def assert_same(z1, z2, names=None):
    for name in names or z1:
        np.testing.assert_allclose(z1[name], z2[name], atol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_question_alone_equals_together(model, variant):
    cfg = config(**variant)
    together, packs = score(model, STATE, QUESTIONS, cfg)
    assert len(packs) == 1
    for name, q in QUESTIONS.items():
        alone, _ = score(model, STATE, {name: q}, cfg)
        assert_same(alone, together, [name])


@pytest.mark.parametrize("variant", VARIANTS)
def test_order_does_not_matter(model, variant):
    cfg = config(**variant)
    z, _ = score(model, STATE, QUESTIONS, cfg)
    shuffled = {
        "d": Choice("pick", ("yy", "x")),
        "b": QUESTIONS["b"],
        "c": Choice("it is {option}.", ("blue", "red", "brown")),
        "a": QUESTIONS["a"],
    }
    z2, _ = score(model, STATE, shuffled, cfg)
    assert_same(z, z2, ["a", "b"])
    np.testing.assert_allclose(z["c"], z2["c"][[1, 2, 0]], atol=1e-5)
    np.testing.assert_allclose(z["d"], z2["d"][[1, 0]], atol=1e-5)


def test_answers_differ_between_questions(model):
    # マスクが効きすぎて質問を見ていない、ということがないこと
    z, _ = score(model, STATE, QUESTIONS, config())
    assert abs(z["a"][0] - z["b"][0]) > 1e-2
    assert np.ptp(z["c"]) > 1e-2


def test_padding_length_does_not_matter(model):
    short, packs = score(model, STATE, QUESTIONS, config())
    long, long_packs = score(model, STATE, QUESTIONS, config(lengths=[256]))
    assert len(packs[0].input_ids) < len(long_packs[0].input_ids) == 256
    assert_same(short, long)


def test_split_passes_equal_single_pass(model):
    single, _ = score(model, STATE, QUESTIONS, config())
    split, packs = score(model, STATE, QUESTIONS, config(max_markers=2))
    assert len(packs) > 1
    assert_same(single, split)


def test_state_truncation(model):
    long_state = STATE + " over the lazy dog and keeps running"
    cfg = config()
    truncated, _ = score(model, long_state, QUESTIONS, cfg)
    same, _ = score(model, long_state[: cfg.max_state_tokens - 2], QUESTIONS, cfg)
    assert_same(truncated, same)


def test_batch_equals_single(model):
    tok = FakeTokenizer()
    cfg = config(lengths=[64])
    packs = [
        pack(tok, STATE, QUESTIONS, cfg)[0],
        pack(tok, "another state", {"b": QUESTIONS["b"], "a": QUESTIONS["a"]}, cfg)[0],
    ]
    with torch.inference_mode():
        singles = [model(**{k: torch.from_numpy(v) for k, v in p.arrays().items()}) for p in packs]
        keys = packs[0].arrays()
        batch = model(
            **{k: torch.from_numpy(np.concatenate([p.arrays()[k] for p in packs])) for k in keys}
        )
    for i, (p, single) in enumerate(zip(packs, singles, strict=True)):
        n = len(p.markers)
        np.testing.assert_allclose(batch[i, :n], single[0, :n], atol=1e-5)


def test_state_matches_original_model(base, model):
    """状態の [CLS] から読んだ値は、状態だけを元のモデルに通した値と一致する (状態は質問を見ない)。

    相対位置 (対数バケット)、マスク、ヘッドの通し方が元のモデルと同じであることの確認。
    """
    state = "a long state " * 3  # 対数バケットに入る距離 (8 以上) を含む長さ
    (p,) = pack(FakeTokenizer(), state, QUESTIONS, config(lengths=[128], max_state_tokens=64))
    inputs = {k: torch.from_numpy(v) for k, v in p.arrays().items()}
    inputs["marker_index"] = torch.zeros_like(inputs["marker_index"])
    n_state = int((p.question_ids == 0).sum())
    with torch.inference_mode():
        z = model(**inputs)[0, 0]
        logits = base(input_ids=inputs["input_ids"][:, :n_state]).logits[0]
    assert n_state > 16
    assert z.item() == pytest.approx((logits[0] - logits[2]).item(), abs=1e-5)


def test_rejects_position_biased_model():
    config = DebertaV2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        relative_attention=True,
        position_biased_input=True,
    )
    with pytest.raises(ValueError):
        PackedModel(DebertaV2ForSequenceClassification(config))
