"""PyTorch バックエンド (NVIDIA GPU / CPU)。

学習済みディレクトリか Hugging Face のモデル ID を読む。
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .scoring import Pair, decision_logits, tokenize


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA が使えません (ワークショップの gpu プラグと setup を確認)")
    return torch.device(device)


class TorchScorer:
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        max_length: int = 256,
        batch_size: int = 32,
        fp16: bool = False,
    ):
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_length = max_length
        self.batch_size = batch_size
        # GTX 16xx など Tensor コアのない GPU では autocast の FP16 のほうが遅いので既定は FP32
        self.fp16 = fp16 and self.device.type == "cuda"

    @torch.inference_mode()
    def score(self, pairs: list[Pair]) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), self.batch_size):
            enc = tokenize(self.tokenizer, pairs[i : i + self.batch_size], self.max_length, "pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
                logits = self.model(**enc).logits
            out.append(logits.float().cpu().numpy())
        return decision_logits(np.concatenate(out), self.id2label)
