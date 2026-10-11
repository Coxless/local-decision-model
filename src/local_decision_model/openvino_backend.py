"""OpenVINO バックエンド (Intel NPU / GPU / CPU)。`export` で作った IR ディレクトリを読む。

NPU は静的な形状でコンパイルするため、入力の形状を固定する。

- ペア方式: [batch_size, max_length]。ペア数が batch_size に満たないときは、空のペアで埋めて
  結果を捨てる
- パック方式: [1, L] (L は推論設定の lengths のどれか)。marker_index は [1, max_markers]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

from .packing import pack, unpack
from .schema import Question
from .scoring import InferenceConfig, Pair, decision_logits, tokenize

IR_FILE = "model.xml"
DEFAULT_CACHE_DIR = ".cache/openvino"


def _read_ir(model_dir: Path) -> ov.Model:
    if not (model_dir / IR_FILE).exists():
        raise FileNotFoundError(
            f"{model_dir / IR_FILE} がありません。GPU 側で `export` してから転送してください"
        )
    return ov.Core().read_model(model_dir / IR_FILE)


def _compile(model: ov.Model, device: str, cache_dir: str | Path | None) -> ov.CompiledModel:
    props: dict[str, str] = {"PERFORMANCE_HINT": "LATENCY"}
    if cache_dir:
        # NPU のコンパイルは数秒〜数十秒かかるので、コンパイル結果をキャッシュする
        props["CACHE_DIR"] = str(cache_dir)
    return ov.Core().compile_model(model, device, props)


def available_devices() -> dict[str, str]:
    core = ov.Core()
    return {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}


class OpenVINOScorer:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "NPU",
        batch_size: int = 8,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    ):
        model_dir = Path(model_dir)
        model = _read_ir(model_dir)
        self.max_length = InferenceConfig.load(model_dir).max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        self.id2label = {int(k): v for k, v in config["id2label"].items()}

        model.reshape({inp.get_any_name(): [batch_size, self.max_length] for inp in model.inputs})
        self.input_names = [inp.get_any_name() for inp in model.inputs]
        self.compiled = _compile(model, device, cache_dir)
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


class OpenVINOPackedScorer:
    """パック方式。系列長ごとに静的な形状でコンパイルし、パックの長さに合うものを使う。

    コンパイルは、その系列長を初めて使うときに行う (`compile_all` でまとめて済ませられる)。
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "NPU",
        config: InferenceConfig | None = None,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        tokenizer=None,
    ):
        """tokenizer を省くと model_dir から読む。"""
        model_dir = Path(model_dir)
        self.model = _read_ir(model_dir)
        self.config = config or InferenceConfig.load(model_dir)
        self.device = device
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_dir)
        self.requests: dict[int, ov.InferRequest] = {}

    def _request(self, length: int) -> ov.InferRequest:
        if length not in self.requests:
            model = self.model.clone()
            model.reshape(
                {
                    inp.get_any_name(): [
                        1,
                        self.config.max_markers if inp.get_any_name() == "marker_index" else length,
                    ]
                    for inp in model.inputs
                }
            )
            compiled = _compile(model, self.device, self.cache_dir)
            self.requests[length] = compiled.create_infer_request()
        return self.requests[length]

    def compile_all(self) -> None:
        """推論設定の lengths をすべてコンパイルする (最初のリクエストを待たせたくないとき)。"""
        for length in self.config.lengths:
            self._request(length)

    def score_questions(self, state: str, questions: dict[str, Question]) -> dict[str, np.ndarray]:
        packs = pack(self.tokenizer, state, questions, self.config)
        outputs = []
        for p in packs:
            request = self._request(len(p.input_ids))
            request.infer(p.arrays())
            # 出力のテンソルは次の推論で上書きされるので、コピーしておく
            outputs.append(request.get_output_tensor(0).data[0].copy())
        return unpack(packs, outputs, questions)
