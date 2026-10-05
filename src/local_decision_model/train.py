"""ファインチューニングと温度較正 (ペア方式とパック方式)。

学習データは「1 行 = 1 つの状態 + 複数の質問 + labels」の JSONL (distill.load_examples)。
labels は 0 / 1 のほか 0〜1 の確率も使える。教師 (Claude Code) の確率をそのまま使えば蒸留になる。

- 損失は質問ごと。noul はソフトラベルの BCE、choice は教師の分布とのクロスエントロピー
- 1 ステップごとに、状態ごとの質問の数 (1〜max_questions)、質問の順番、選択肢の順番を乱数で変える
- 単語埋め込みは凍結し、fp32 で学習する。1 ステップ (batch_size 件の状態) は、max_tokens に
  収まる大きさに分けて順伝播し、勾配を累積する
- 終わりに検証データで noul と choice の温度を別々に求め、推論設定に保存する

ペア方式とパック方式で、データの選び方と損失は同じ。違うのは系列の作り方だけなので、
同じ学習データでの比較に使える。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .decide import Decider
from .distill import Example, load_examples
from .evaluate import calibrate, metrics, score
from .packed_model import PackedModel
from .packing import pack
from .schema import Choice, Question, parse_question
from .scoring import InferenceConfig, decision_logits
from .torch_backend import TorchPackedScorer, TorchScorer, resolve_device

# 質問ごとの (判断ロジットの位置, 教師の確率)。noul は 1 個、choice は選択肢の数だけ並ぶ
Target = tuple[list[int], list[float]]


def split(
    examples: list[Example], val_ratio: float, seed: int
) -> tuple[list[Example], list[Example]]:
    examples = examples[:]
    random.Random(seed).shuffle(examples)
    n_val = max(1, int(len(examples) * val_ratio))
    return examples[n_val:], examples[:n_val]


def sample_questions(
    example: Example, rng: random.Random, max_questions: int | None = None
) -> tuple[dict[str, Question], dict[str, list[float]]]:
    """質問を 1〜max_questions 個選び、質問と選択肢の順番を入れ替える。

    (質問, 質問名 → 教師の確率) を返す。確率は、noul が 1 個、choice が入れ替えた後の選択肢の順。
    """
    names = list(example.questions)
    rng.shuffle(names)
    limit = min(max_questions or len(names), len(names))
    questions: dict[str, Question] = {}
    probs: dict[str, list[float]] = {}
    for name in names[: rng.randint(1, limit)]:
        q = parse_question(example.questions[name])
        if isinstance(q, Choice):
            options = list(q.options)
            rng.shuffle(options)
            q = Choice(q.instructions, tuple(options))
            probs[name] = [example.labels[name][o] for o in options]
        else:
            probs[name] = [example.labels[name]]
        questions[name] = q
    return questions, probs


def question_loss(z: torch.Tensor, targets: list[Target]) -> torch.Tensor:
    """質問ごとの損失の合計。z は判断ロジットを 1 列に並べたもの。"""
    noul = [(index[0], probs[0]) for index, probs in targets if len(index) == 1]
    loss = z.new_zeros(())
    if noul:
        index, y = zip(*noul, strict=True)
        loss = loss + F.binary_cross_entropy_with_logits(
            z[list(index)], z.new_tensor(y), reduction="sum"
        )
    for index, probs in targets:
        if len(index) > 1:
            loss = loss - (z.new_tensor(probs) * F.log_softmax(z[index], dim=0)).sum()
    return loss


@dataclass
class _Item:
    """まとめて順伝播する単位。rows は系列 (1 行 = 1 本)、targets の位置は rows の中での番号。"""

    rows: list[dict[str, np.ndarray]]
    targets: list[Target]

    @property
    def length(self) -> int:
        return max(len(row["input_ids"]) for row in self.rows)


class _PairArch:
    """ペア方式: (状態, 仮説) ごとに 1 本の系列。判断ロジットは 1 本につき 1 個。"""

    width = 1

    def __init__(self, model, tokenizer, max_length: int):
        self.model, self.base = model, model
        self.tokenizer, self.max_length = tokenizer, max_length
        self.pad = {"input_ids": tokenizer.pad_token_id, "attention_mask": 0, "token_type_ids": 0}
        self.id2label = {int(k): v for k, v in model.config.id2label.items()}

    def items(
        self, state: str, questions: dict[str, Question], probs: dict[str, list[float]]
    ) -> list[_Item]:
        """質問ごとに 1 つ (ペア方式では、質問が違えば別々の順伝播に分けてよい)。"""
        items = []
        for name, q in questions.items():
            hyps = q.hypotheses()
            enc = self.tokenizer(
                [state] * len(hyps), hyps, truncation="only_first", max_length=self.max_length
            )
            rows = [
                {key: np.asarray(values[i]) for key, values in enc.items() if key in self.pad}
                for i in range(len(hyps))
            ]
            items.append(_Item(rows, [(list(range(len(hyps))), probs[name])]))
        return items

    def logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return decision_logits(self.model(**inputs).logits, self.id2label)


class _PackedArch:
    """パック方式: 状態と質問を 1 本に詰める。判断ロジットは 1 本につき max_markers 個。"""

    def __init__(self, model: PackedModel, tokenizer, config: InferenceConfig):
        self.model, self.base = model, model.base
        self.tokenizer, self.config = _CachedTokenizer(tokenizer), config
        self.width = config.max_markers
        self.pad = {
            "input_ids": tokenizer.pad_token_id,
            "question_ids": -1,
            "option_ids": 0,
            "positions": 0,
            "marker_index": 0,
        }

    def items(
        self, state: str, questions: dict[str, Question], probs: dict[str, list[float]]
    ) -> list[_Item]:
        """状態ごとに 1 つ (choice の選択肢が別のパックに分かれても、同じ順伝播に入れる)。"""
        packs = pack(self.tokenizer, state, questions, self.config)
        index: dict[str, dict[int, int]] = {}
        for row, p in enumerate(packs):
            for marker_slot, (name, option) in enumerate(p.markers):
                index.setdefault(name, {})[option] = row * self.width + marker_slot
        targets = [
            ([index[name][i] for i in range(len(probs[name]))], probs[name]) for name in questions
        ]
        return [_Item([{key: getattr(p, key) for key in self.pad} for p in packs], targets)]

    def logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**inputs).reshape(-1)


class _CachedTokenizer:
    """pack が同じ質問や状態を毎ステップ トークナイズし直さないようにする。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cache: dict[str, dict] = {}

    def __call__(self, text: str, add_special_tokens: bool = False):
        if text not in self.cache:
            self.cache[text] = self.tokenizer(text, add_special_tokens=add_special_tokens)
        return self.cache[text]


def _chunks(items: list[_Item], max_tokens: int) -> list[list[_Item]]:
    """パディング後のトークン数 (系列の数 × 最長) が max_tokens に収まるように分ける。

    1 つの _Item の系列は分けない (choice の選択肢が別々の順伝播に分かれないようにする)。
    """
    chunks: list[list[_Item]] = []
    rows = length = 0
    for item in sorted(items, key=lambda item: item.length):
        rows, length = rows + len(item.rows), max(length, item.length)
        if not chunks or rows * length > max_tokens:
            chunks.append([])
            rows, length = len(item.rows), item.length
        chunks[-1].append(item)
    return chunks


def _collate(
    chunk: list[_Item], arch: _PairArch | _PackedArch, device: torch.device
) -> tuple[dict[str, torch.Tensor], list[Target]]:
    rows = [row for item in chunk for row in item.rows]
    inputs = {}
    for key in rows[0]:
        length = max(len(row[key]) for row in rows)
        batch = np.full((len(rows), length), arch.pad[key], dtype=np.int64)
        for i, row in enumerate(rows):
            batch[i, : len(row[key])] = row[key]
        inputs[key] = torch.from_numpy(batch).to(device)
    targets, offset = [], 0
    for item in chunk:
        targets += [([offset + i for i in index], probs) for index, probs in item.targets]
        offset += len(item.rows) * arch.width
    return inputs, targets


def fit(
    arch: _PairArch | _PackedArch,
    examples: list[Example],
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-5,
    max_questions: int | None = None,
    max_tokens: int = 2048,
    seed: int = 0,
    fp16: bool = False,
    log_every: int = 50,
) -> list[float]:
    """arch.model を学習する。エポックごとの、質問 1 つあたりの損失を返す。"""
    model = arch.model
    arch.base.get_input_embeddings().weight.requires_grad_(False)
    # 層ごとの途中結果を持たずに逆伝播のときに計算し直す (6 GB の GPU に収めるため)
    arch.base.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(examples) / batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    rng = random.Random(seed)
    examples = examples[:]
    losses = []

    step = 0
    for epoch in range(epochs):
        model.train()
        rng.shuffle(examples)
        total = count = 0.0
        for start in range(0, len(examples), batch_size):
            items = [
                item
                for e in examples[start : start + batch_size]
                for item in arch.items(e.state, *sample_questions(e, rng, max_questions))
            ]
            n_questions = sum(len(item.targets) for item in items)
            opt.zero_grad(set_to_none=True)
            for chunk in _chunks(items, max_tokens):
                inputs, targets = _collate(chunk, arch, device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    z = arch.logits(inputs)
                loss = question_loss(z.float(), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("loss が NaN / inf になりました")
                scaler.scale(loss / n_questions).backward()
                total += loss.item()
            count += n_questions
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % log_every == 0:
                print(f"step {step}/{steps}  train_loss={total / count:.4f}", flush=True)
        losses.append(total / count)
        print(f"epoch {epoch + 1}/{epochs}  train_loss={losses[-1]:.4f}", flush=True)
    return losses


def train(
    data: str | Path,
    out_dir: str | Path,
    base_model: str,
    arch: str = "pair",
    device: str = "cuda",
    val_data: str | Path | None = None,
    eval_data: str | Path | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-5,
    max_length: int = 256,
    max_questions: int | None = None,
    max_tokens: int = 2048,
    state_sees_questions: bool = False,
    val_ratio: float = 0.1,
    seed: int = 0,
    fp16: bool = False,
) -> dict:
    """学習して out_dir に保存し、検証データで温度を較正する。指標を返す。

    eval_data があれば、較正した温度のまま評価セットの指標も出す。
    """
    torch.manual_seed(seed)
    dev = resolve_device(device)
    examples = load_examples(data)
    if val_data:
        train_examples, val_examples = examples, load_examples(val_data)
    else:
        train_examples, val_examples = split(examples, val_ratio, seed)

    config = InferenceConfig(
        arch=arch, max_length=max_length, state_sees_questions=state_sees_questions
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if arch == "packed":
        model = PackedModel.from_pretrained(base_model, state_sees_questions).to(dev)
        trainer = _PackedArch(model, tokenizer, config)
    else:
        # fp16 の重みでも fp32 で学習する (mDeBERTa は fp16 だと NaN が出やすい)
        model = AutoModelForSequenceClassification.from_pretrained(base_model, dtype=torch.float32)
        trainer = _PairArch(model.to(dev), tokenizer, max_length)

    losses = fit(
        trainer,
        train_examples,
        dev,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_questions=max_questions,
        max_tokens=max_tokens,
        seed=seed,
        fp16=fp16,
    )

    out_dir = Path(out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    config.save(out_dir)
    # 保存したものを読み直して推論する (推論のときと同じ経路で温度を求める)
    del model, trainer
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if arch == "packed":
        scorer = TorchPackedScorer(str(out_dir), device, config)
    else:
        scorer = TorchScorer(str(out_dir), device, max_length)
    decider = Decider(scorer)

    val_scores = score(decider, val_examples)
    val_scores.save(out_dir / "val_scores.npz")
    config.temperature_noul, config.temperature_choice = calibrate(val_scores)
    config.save(out_dir)

    def both(scores) -> dict:
        calibrated = metrics(scores, config.temperature_noul, config.temperature_choice)
        return {"uncalibrated": metrics(scores), "calibrated": calibrated}

    report = {
        "arch": arch,
        "train_loss": losses,
        "temperature_noul": config.temperature_noul,
        "temperature_choice": config.temperature_choice,
        "val": both(val_scores),
    }
    if eval_data:
        eval_scores = score(decider, load_examples(eval_data))
        eval_scores.save(out_dir / "eval_scores.npz")
        report["eval"] = both(eval_scores)
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
