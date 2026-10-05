"""学習のテスト。小さなランダム初期化の DeBERTa を CPU で動かす。"""

import random

import numpy as np
import pytest

from local_decision_model.distill import Example
from local_decision_model.packing import pack, unpack
from local_decision_model.schema import Choice
from local_decision_model.scoring import InferenceConfig

torch = pytest.importorskip("torch")

from transformers import DebertaV2Config, DebertaV2ForSequenceClassification  # noqa: E402

from local_decision_model.packed_model import PackedModel  # noqa: E402
from local_decision_model.train import (  # noqa: E402
    _chunks,
    _collate,
    _Item,
    _PackedArch,
    fit,
    question_loss,
    sample_questions,
)

VOCAB = 200


class FakeTokenizer:
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + ord(c) % (VOCAB - 3) for c in text]}


def packed_model():
    torch.manual_seed(0)
    config = DebertaV2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        relative_attention=True,
        position_buckets=16,
        pos_att_type=["p2c", "c2p"],
        share_att_key=True,
        norm_rel_ebd="layer_norm",
        position_biased_input=False,
        type_vocab_size=0,
        initializer_range=0.3,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )
    return PackedModel(DebertaV2ForSequenceClassification(config))


QUESTIONS = {
    "quick": "is it quick",
    "dog": "is it a dog",
    "color": {"type": "choice", "instructions": "it is {option}.", "options": ["red", "brown"]},
    "size": {"type": "choice", "instructions": "pick", "options": ["s", "m", "l"]},
}
EXAMPLES = [
    Example(
        "the quick brown fox",
        QUESTIONS,
        {
            "quick": 0.9,
            "dog": 0.1,
            "color": {"red": 0.1, "brown": 0.9},
            "size": {"s": 0.2, "m": 0.7, "l": 0.1},
        },
    ),
    Example(
        "a slow red dog",
        QUESTIONS,
        {
            "quick": 0.0,
            "dog": 1.0,
            "color": {"red": 1.0, "brown": 0.0},
            "size": {"s": 0.1, "m": 0.1, "l": 0.8},
        },
    ),
]


def config(**kwargs):
    defaults = dict(arch="packed", lengths=[32, 64, 128], max_markers=8, max_state_tokens=24)
    return InferenceConfig(**{**defaults, **kwargs})


def test_sample_questions_keeps_labels_aligned():
    example = EXAMPLES[0]
    seen_counts, seen_orders = set(), set()
    for seed in range(50):
        questions, probs = sample_questions(example, random.Random(seed))
        assert questions.keys() == probs.keys() <= QUESTIONS.keys()
        seen_counts.add(len(questions))
        seen_orders.add(tuple(questions))
        for name, q in questions.items():
            if isinstance(q, Choice):
                assert probs[name] == [example.labels[name][o] for o in q.options]
            else:
                assert probs[name] == [example.labels[name]]
    assert seen_counts == {1, 2, 3, 4}
    assert len(seen_orders) > 10


def test_sample_questions_respects_max_questions():
    for seed in range(20):
        questions, _ = sample_questions(EXAMPLES[0], random.Random(seed), max_questions=2)
        assert 1 <= len(questions) <= 2


def test_question_loss_is_bce_plus_cross_entropy():
    z = torch.tensor([0.5, -1.0, 2.0, 0.0, 1.0])
    targets = [([1], [0.2]), ([0, 2, 4], [0.1, 0.6, 0.3]), ([3], [1.0])]
    bce = sum(
        -(y * np.log(p) + (1 - y) * np.log(1 - p)) for y, p in [(0.2, 1 / (1 + np.e)), (1.0, 0.5)]
    )
    s = np.array([0.5, 2.0, 1.0])
    ce = -np.sum(np.array([0.1, 0.6, 0.3]) * (s - np.log(np.exp(s).sum())))
    assert question_loss(z, targets).item() == pytest.approx(bce + ce, rel=1e-5)


def test_chunks_keep_items_whole_and_within_budget():
    def item(rows, length):
        return _Item([{"input_ids": np.zeros(length)}] * rows, [])

    items = [item(2, 64), item(1, 32), item(3, 128), item(1, 64)]
    chunks = _chunks(items, max_tokens=256)
    assert sorted(id(i) for c in chunks for i in c) == sorted(id(i) for i in items)
    for chunk in chunks:
        rows = sum(len(i.rows) for i in chunk)
        assert len(chunk) == 1 or rows * max(i.length for i in chunk) <= 256


@pytest.mark.parametrize("max_markers", [8, 3])
def test_packed_targets_point_at_the_right_logits(max_markers):
    """学習で読む判断ロジットの位置が、推論 (pack → unpack) と同じ値を指している。"""
    model = packed_model().eval()
    cfg = config(max_markers=max_markers)
    arch = _PackedArch(model, FakeTokenizer(), cfg)
    items, expected = [], []
    for seed, example in enumerate(EXAMPLES):
        questions, probs = sample_questions(example, random.Random(seed))
        items += arch.items(example.state, questions, probs)
        packs = pack(FakeTokenizer(), example.state, questions, cfg)
        with torch.inference_mode():
            outputs = [
                model(**{k: torch.from_numpy(v) for k, v in p.arrays().items()})[0].numpy()
                for p in packs
            ]
        z = unpack(packs, outputs, questions)
        expected += [z[name] for name in questions]

    inputs, targets = _collate(items, arch, torch.device("cpu"))
    with torch.inference_mode():
        z = arch.logits(inputs).numpy()
    assert len(targets) == len(expected)
    for (index, _), want in zip(targets, expected, strict=True):
        np.testing.assert_allclose(z[index], want, atol=1e-5)


def test_fit_reduces_loss_and_freezes_word_embeddings():
    model = packed_model()
    words = model.base.get_input_embeddings().weight.clone()
    arch = _PackedArch(model, FakeTokenizer(), config())
    losses = fit(arch, EXAMPLES, torch.device("cpu"), epochs=30, batch_size=2, lr=1e-2)
    assert losses[-1] < losses[0] * 0.7
    assert torch.equal(model.base.get_input_embeddings().weight, words)
    assert model.role_embeddings.weight.abs().sum() > 0
