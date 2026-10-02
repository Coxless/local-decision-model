"""コマンドライン: python -m local_decision_model <devices|decide|bench|train|export>

torch / openvino はそれぞれの extra を入れたワークショップでしか使えないので、
必要なときだけ import する。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

from .decide import Decider
from .schema import parse_questions
from .scoring import DEFAULT_BASE_MODEL, S1Config

DEFAULT_IR_DIR = "models/base-ov"


def make_scorer(args):
    """(scorer, 温度) を返す。"""
    if args.backend == "torch":
        from .torch_backend import TorchScorer

        model = args.model or DEFAULT_BASE_MODEL
        config = S1Config.load(model) if Path(model).is_dir() else S1Config()
        return TorchScorer(model, args.device, config.max_length), config.temperature
    from .openvino_backend import OpenVINOScorer

    model = args.model or DEFAULT_IR_DIR
    return OpenVINOScorer(model, args.device, args.batch_size), S1Config.load(model).temperature


def load_request(path: str) -> tuple[str, dict]:
    with open(path, encoding="utf-8") as f:
        req = yaml.safe_load(f)
    return req["state"], parse_questions(req["questions"])


def cmd_devices(_args) -> None:
    try:
        import torch

        cuda = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "なし"
        print(f"torch {torch.__version__}  CUDA: {cuda}")
    except ImportError:
        print("torch: 未インストール (cuda extra)")
    try:
        import openvino

        from .openvino_backend import available_devices

        print(f"openvino {openvino.__version__}")
        for dev, name in available_devices().items():
            print(f"  {dev}: {name}")
    except ImportError:
        print("openvino: 未インストール (openvino extra)")


def cmd_decide(args) -> None:
    state, questions = load_request(args.request)
    scorer, temperature = make_scorer(args)
    decider = Decider(scorer, temperature)
    start = time.perf_counter()
    answers = decider.decide(state, questions)
    elapsed = (time.perf_counter() - start) * 1000
    out = {"answers": {k: a.to_dict() for k, a in answers.items()}, "latency_ms": round(elapsed, 2)}
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    print()


def cmd_bench(args) -> None:
    state, questions = load_request(args.request)
    scorer, temperature = make_scorer(args)
    decider = Decider(scorer, temperature)
    for _ in range(args.warmup):
        decider.decide(state, questions)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        decider.decide(state, questions)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    print(
        f"{args.backend}/{args.device}  runs={args.runs}  "
        f"p50={statistics.median(times):.2f}ms  p95={times[int(0.95 * (len(times) - 1))]:.2f}ms  "
        f"min={times[0]:.2f}ms"
    )


def cmd_train(args) -> None:
    from .train import train

    report = train(
        args.data,
        args.out,
        args.base_model,
        device=args.device,
        val_data=args.val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        fp16=args.fp16,
    )
    print(json.dumps(report, indent=2))


def cmd_export(args) -> None:
    from .export import export

    out = export(args.model, args.out, fp16=not args.fp32)
    print(f"OpenVINO IR を書き出しました: {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="local_decision_model")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="使えるデバイスを表示する").set_defaults(func=cmd_devices)

    for name, func, help_ in [
        ("decide", cmd_decide, "リクエスト YAML に回答する"),
        ("bench", cmd_bench, "レイテンシを計測する"),
    ]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("request", nargs="?", default="examples/review.yaml")
        s.add_argument("--backend", choices=["torch", "openvino"], default="torch")
        s.add_argument("--device", default="cuda", help="torch: cuda/cpu, openvino: NPU/GPU/CPU")
        s.add_argument(
            "--model",
            help=f"torch: HF ID か学習済みディレクトリ (既定 {DEFAULT_BASE_MODEL}), "
            f"openvino: IR ディレクトリ (既定 {DEFAULT_IR_DIR})",
        )
        s.add_argument("--batch-size", type=int, default=8, help="openvino の静的バッチサイズ")
        s.set_defaults(func=func)
        if name == "bench":
            s.add_argument("--runs", type=int, default=100)
            s.add_argument("--warmup", type=int, default=10)

    s = sub.add_parser("train", help="JSONL でファインチューニングし、温度を較正する")
    s.add_argument("data")
    s.add_argument("--out", default="models/finetuned")
    s.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    s.add_argument("--val-data")
    s.add_argument("--device", default="cuda")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--lr", type=float, default=3e-5)
    s.add_argument("--max-length", type=int, default=256)
    s.add_argument(
        "--fp16", action="store_true", help="混合精度で学習する (Tensor コアのある GPU 向け)"
    )
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("export", help="OpenVINO IR に変換する (NPU 用)")
    s.add_argument("--model", default=DEFAULT_BASE_MODEL, help="HF ID か学習済みディレクトリ")
    s.add_argument("--out", default=DEFAULT_IR_DIR)
    s.add_argument("--fp32", action="store_true", help="重みを FP16 に圧縮しない")
    s.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    args.func(args)
