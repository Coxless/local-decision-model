"""PyTorch モデル → OpenVINO IR。出力ディレクトリだけで NPU 側の推論ができるようにする
(IR + トークナイザ + config.json + s1.json)。torch が要るので GPU 側のワークショップで実行する。
"""

from __future__ import annotations

from pathlib import Path

import openvino as ov
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .openvino_backend import IR_FILE
from .scoring import InferenceConfig, tokenize


class _LogitsOnly(torch.nn.Module):
    """HF モデルの出力 (dataclass) を logits テンソルだけにする。"""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model: str, out_dir: str | Path, fp16: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = InferenceConfig.load(model) if Path(model).is_dir() else InferenceConfig()

    tokenizer = AutoTokenizer.from_pretrained(model)
    # SDPA はトレース時に分岐が固定されるので、素直な eager attention で変換する
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        model, attn_implementation="eager", dtype=torch.float32
    ).eval()

    pairs = [("example state", "example hypothesis")]
    example = tokenize(tokenizer, pairs, config.max_length, "pt")
    names = ["input_ids", "attention_mask"]
    with torch.no_grad():
        ov_model = ov.convert_model(
            _LogitsOnly(hf_model), example_input=tuple(example[n] for n in names)
        )
    for inp, name in zip(ov_model.inputs, names, strict=True):
        inp.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({"logits"})
    # 形状は動的のまま保存し、読み込み側 (NPU) で静的に reshape する
    ov.save_model(ov_model, out_dir / IR_FILE, compress_to_fp16=fp16)

    tokenizer.save_pretrained(out_dir)
    hf_model.config.save_pretrained(out_dir)
    config.save(out_dir)
    return out_dir
