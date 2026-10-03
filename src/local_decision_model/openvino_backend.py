"""OpenVINO バックエンド (Intel NPU / GPU / CPU)。`export` で作った IR ディレクトリを読む。

NPU は静的な形状でコンパイルするため、入力を [batch_size, max_length] に固定する。
ペア数が batch_size に満たないときは空のペアで埋めて結果を捨てる。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

from .scoring import InferenceConfig, Pair, decision_logits, tokenize

IR_FILE = "model.xml"


def available_devices() -> dict[str, str]:
    core = ov.Core()
    return {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}


class OpenVINOScorer:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "NPU",
        batch_size: int = 8,
        cache_dir: str | Path | None = ".cache/openvino",
    ):
        model_dir = Path(model_dir)
        if not (model_dir / IR_FILE).exists():
            raise FileNotFoundError(
                f"{model_dir / IR_FILE} がありません。GPU 側で `export` してから転送してください"
            )
        self.max_length = InferenceConfig.load(model_dir).max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        self.id2label = {int(k): v for k, v in config["id2label"].items()}

        core = ov.Core()
        model = core.read_model(model_dir / IR_FILE)
        model.reshape({inp.get_any_name(): [batch_size, self.max_length] for inp in model.inputs})
        self.input_names = [inp.get_any_name() for inp in model.inputs]
        props: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
        if cache_dir:
            # NPU のコンパイルは数秒〜数十秒かかるので、コンパイル結果をキャッシュする
            props["CACHE_DIR"] = str(cache_dir)
        self.compiled = core.compile_model(model, device, props)
        self.request = self.compiled.create_infer_request()

    def score(self, pairs: list[Pair]) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i : i + self.batch_size]
            padded = chunk + [("", "")] * (self.batch_size - len(chunk))
            enc = tokenize(self.tokenizer, padded, self.max_length)
            inputs = {name: enc[name].astype(np.int64) for name in self.input_names}
            result = self.request.infer(inputs)
            out.append(result[self.compiled.output(0)][: len(chunk)])
        return decision_logits(np.concatenate(out), self.id2label)
