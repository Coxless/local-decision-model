"""IR に含まれる演算の種類と数を一覧にする。

NPU でコンパイルできなかったときに、原因の演算を探す手がかりにする。NPU と同じ静的な形状にして、
定数の畳み込みをした後の演算を数える (形状の計算は畳み込まれて消える)。

    uv run python scripts/phase4/ir_ops.py models/packed-ov models/mdeberta-ov
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import openvino as ov
from openvino.passes import ConstantFolding, Manager

from local_decision_model.openvino_backend import IR_FILE
from local_decision_model.scoring import InferenceConfig


def static_model(model_dir: str) -> ov.Model:
    config = InferenceConfig.load(model_dir)
    model = ov.Core().read_model(Path(model_dir) / IR_FILE)
    if config.arch == "packed":
        shapes = {"marker_index": [1, config.max_markers]}
        default = [1, max(config.lengths)]
    else:
        shapes, default = {}, [8, config.max_length]
    names = [inp.get_any_name() for inp in model.inputs]
    model.reshape({name: shapes.get(name, default) for name in names})
    manager = Manager()
    manager.register_pass(ConstantFolding())
    manager.run_passes(model)
    return model


def main(model_dirs: list[str]) -> None:
    models = [static_model(d) for d in model_dirs]
    counts = [
        Counter(
            f"{op.get_type_name()} (opset{op.get_type_info().version_id[5:]})"
            for op in model.get_ordered_ops()
        )
        for model in models
    ]
    print("| 演算 | " + " | ".join(model_dirs) + " |")
    print("|---|" + "---|" * len(model_dirs))
    for name in sorted(set().union(*counts)):
        print(f"| {name} | " + " | ".join(str(c[name] or "") for c in counts) + " |")

    for model_dir, model in zip(model_dirs, models, strict=True):
        ops = model.get_ordered_ops()
        dynamic = sum(any(o.get_partial_shape().is_dynamic for o in op.outputs()) for op in ops)
        # 条件が定数の If は、コンパイル時にどちらかの枝に置き換えられる
        branching = sum(
            op.get_type_name() == "If"
            and op.input_value(0).get_node().get_type_name() != "Constant"
            for op in ops
        )
        print(f"{model_dir}: 形状が動的な演算 {dynamic}、条件が定数でない If {branching}")


if __name__ == "__main__":
    main(sys.argv[1:])
