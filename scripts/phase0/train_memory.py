"""フェーズ 0: 6 GB の GPU で mDeBERTa の fp32 学習が回るかを測る。

単語埋め込みの凍結 + gradient checkpointing で、バッチサイズと系列長ごとに
ピークメモリと 1 ステップの時間を出す。

    workshop exec gpu -- bash -c 'cd /project && uv run python scripts/phase0/train_memory.py'
"""

from __future__ import annotations

import itertools
import time

import torch
from transformers import AutoModelForSequenceClassification

MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
STEPS = 5


def measure(model, optimizer, batch: int, length: int) -> tuple[float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(5, 250000, (batch, length), device="cuda")
    mask = torch.ones_like(x)
    target = torch.rand(batch, device="cuda")
    times = []
    for _ in range(STEPS + 1):
        torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(input_ids=x, attention_mask=mask).logits
        z = logits[:, 0] - logits[:, 2]  # entailment - contradiction
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if not torch.isfinite(loss):
            raise RuntimeError("loss が NaN / inf になりました")
    return torch.cuda.max_memory_allocated() / 2**30, sum(times[1:]) / STEPS * 1000


def main() -> None:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float32).cuda()
    model.train()
    model.get_input_embeddings().weight.requires_grad_(False)
    model.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in params)
    print(f"params: total {total / 1e6:.0f}M, trainable {trainable / 1e6:.0f}M")
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    print(f"weights+optimizer (before step): {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    print("batch x length  peak GiB  ms/step  tokens/s")
    for batch, length in itertools.chain(
        [(b, 128) for b in (8, 16)], [(b, 256) for b in (4, 8)], [(b, 512) for b in (2, 4)]
    ):
        try:
            peak, ms = measure(model, optimizer, batch, length)
            tps = batch * length / ms * 1000
            print(f"{batch:>5} x {length:<6}  {peak:8.2f}  {ms:7.0f}  {tps:8.0f}")
        except torch.OutOfMemoryError:
            print(f"{batch:>5} x {length:<6}  OOM")
            optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
