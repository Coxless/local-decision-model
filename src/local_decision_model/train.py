"""ファインチューニングと温度較正。

データは JSONL で、1 行 1 ペア:
    {"state": "...", "question": "仮説文 (Noul の instructions)", "label": 0 か 1 (0〜1 の確率も可)}

Choice の質問も「instructions + 選択肢」の仮説文に展開されるので、同じ形式で学習できる
(正解の選択肢を 1、それ以外を 0 にする)。ソフトラベルを使えば、大きな LLM が出した確率を
そのまま教師にした蒸留にもなる。
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .calibration import binary_metrics, fit_temperature
from .scoring import InferenceConfig, Pair, decision_logits, tokenize
from .torch_backend import resolve_device

Example = tuple[Pair, float]


def load_jsonl(path: str | Path) -> list[Example]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append(((r["state"], r["question"]), float(r["label"])))
    return rows


def split(rows: list[Example], val_ratio: float, seed: int) -> tuple[list, list]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]


def _decision_logits_t(logits: torch.Tensor, id2label: dict[int, str]) -> torch.Tensor:
    # scoring.decision_logits の torch 版 (勾配を通すため)
    labels = {v.lower(): k for k, v in id2label.items()}
    if logits.shape[-1] == 1:
        return logits[:, 0]
    if "entailment" in labels and "contradiction" in labels:
        return logits[:, labels["entailment"]] - logits[:, labels["contradiction"]]
    return logits[:, 1] - logits[:, 0]


@torch.inference_mode()
def predict_logits(model, tokenizer, rows, max_length, device, batch_size=64) -> np.ndarray:
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    out = []
    for i in range(0, len(rows), batch_size):
        enc = tokenize(tokenizer, [p for p, _ in rows[i : i + batch_size]], max_length, "pt")
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        out.append(logits.float().cpu().numpy())
    return decision_logits(np.concatenate(out), id2label)


def train(
    data: str | Path,
    out_dir: str | Path,
    base_model: str,
    device: str = "cuda",
    val_data: str | Path | None = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 3e-5,
    max_length: int = 256,
    val_ratio: float = 0.1,
    seed: int = 0,
    fp16: bool = False,
) -> dict[str, dict[str, float]]:
    torch.manual_seed(seed)
    dev = resolve_device(device)
    rows = load_jsonl(data)
    if val_data:
        train_rows, val_rows = rows, load_jsonl(val_data)
    else:
        train_rows, val_rows = split(rows, val_ratio, seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # fp16 の重みでも fp32 で学習する (mDeBERTa は fp16 だと NaN が出やすい)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, dtype=torch.float32).to(
        dev
    )
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    y_val = np.array([y for _, y in val_rows])
    z_before = predict_logits(model, tokenizer, val_rows, max_length, dev)
    report = {"before": binary_metrics(z_before, y_val)}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train_rows) / batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = fp16 and dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = random.Random(seed)

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_rows)
        total = 0.0
        for i in range(0, len(train_rows), batch_size):
            batch = train_rows[i : i + batch_size]
            enc = tokenize(tokenizer, [p for p, _ in batch], max_length, "pt")
            y = torch.tensor([t for _, t in batch], device=dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
            loss = loss_fn(_decision_logits_t(logits.float(), id2label), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}/{epochs}  train_loss={total / len(train_rows):.4f}")

    z_val = predict_logits(model, tokenizer, val_rows, max_length, dev)
    temperature = fit_temperature(z_val, y_val)
    report["after"] = binary_metrics(z_val, y_val)
    report["calibrated"] = binary_metrics(z_val, y_val, temperature)
    report["calibrated"]["temperature"] = temperature

    out_dir = Path(out_dir)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    InferenceConfig(
        max_length=max_length, temperature_noul=temperature, temperature_choice=temperature
    ).save(out_dir)
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
