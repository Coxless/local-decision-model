"""PyTorch モデル → OpenVINO IR。出力ディレクトリだけで NPU 側の推論ができるようにする
(IR + トークナイザ + config.json + s1.json)。torch が要るので GPU 側のワークショップで実行する。

方式はモデルディレクトリの s1.json で決まる。パック方式の IR は、役割埋め込み、
マスクと相対位置の計算、判断ロジットへの変換までを含む。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .openvino_backend import IR_FILE
from .packed_model import PackedModel
from .packing import pack
from .schema import Noul
from .scoring import InferenceConfig, tokenize

PACKED_OUTPUT = "decision_logits"


class _LogitsOnly(torch.nn.Module):
    """HF モデルの出力 (dataclass) を logits テンソルだけにする。"""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def _convert(module: torch.nn.Module, example: dict, output: str) -> ov.Model:
    names = list(example)
    with torch.no_grad():
        ov_model = ov.convert_model(module, example_input=tuple(example[n] for n in names))
    for inp, name in zip(ov_model.inputs, names, strict=True):
        inp.get_tensor().set_names({name})
    ov_model.outputs[0].get_tensor().set_names({output})
    return ov_model


def convert_packed(model: PackedModel, example: dict[str, np.ndarray]) -> ov.Model:
    """パック方式のモデルを IR にする。example は Pack.arrays() (トレースに使う入力の例)。

    入力は input_ids / question_ids / option_ids / positions が [B, L]、marker_index が [B, M]、
    出力は判断ロジット [B, M]。
    """
    inputs = {name: torch.from_numpy(value) for name, value in example.items()}
    return _convert(model.eval(), inputs, PACKED_OUTPUT)


def export(model: str, out_dir: str | Path, fp16: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = InferenceConfig.load(model) if Path(model).is_dir() else InferenceConfig()

    tokenizer = AutoTokenizer.from_pretrained(model)
    if config.arch == "packed":
        packed = PackedModel.from_pretrained(model, config.state_sees_questions)
        (example,) = pack(tokenizer, "example state", {"q": Noul("example hypothesis")}, config)
        ov_model = convert_packed(packed, example.arrays())
        hf_config = packed.base.config
    else:
        # SDPA はトレース時に分岐が固定されるので、素直な eager attention で変換する
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            model, attn_implementation="eager", dtype=torch.float32
        ).eval()
        pairs = [("example state", "example hypothesis")]
        enc = tokenize(tokenizer, pairs, config.max_length, "pt")
        example = {name: enc[name] for name in ("input_ids", "attention_mask")}
        ov_model = _convert(_LogitsOnly(hf_model), example, "logits")
        hf_config = hf_model.config
    # 形状は動的のまま保存し、読み込み側 (NPU) で静的に reshape する
    ov.save_model(ov_model, out_dir / IR_FILE, compress_to_fp16=fp16)

    tokenizer.save_pretrained(out_dir)
    hf_config.save_pretrained(out_dir)
    config.save(out_dir)
    return out_dir
