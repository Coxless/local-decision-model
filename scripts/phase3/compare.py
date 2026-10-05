"""フェーズ 3: 学習したモデルを、検証データと評価セットで比べる。

モデルディレクトリごとに、検証データで温度を求め直して推論設定 (s1.json) に書き、
評価セットの指標、質問ごとの正解率、モデルどうしの判定の一致を Markdown の表で出す。
判断ロジット (val_scores.npz / eval_scores.npz) がモデルディレクトリになければ、推論して作る。

    workshop exec gpu -- bash -c 'cd /project && uv run python scripts/phase3/compare.py \
        models/packed models/packed-all models/pair'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from local_decision_model.decide import Decider
from local_decision_model.distill import load_examples
from local_decision_model.evaluate import Scores, calibrate, metrics, score
from local_decision_model.scoring import InferenceConfig
from local_decision_model.train import split


def load_scores(model_dir: Path, name: str, examples, device: str) -> Scores:
    path = model_dir / f"{name}_scores.npz"
    if not path.exists():
        from local_decision_model.torch_backend import TorchPackedScorer, TorchScorer

        config = InferenceConfig.load(model_dir)
        if config.arch == "packed":
            scorer = TorchPackedScorer(str(model_dir), device, config)
        else:
            scorer = TorchScorer(str(model_dir), device, config.max_length)
        score(Decider(scorer), examples).save(path)
    return Scores.load(path)


def decisions(scores: Scores) -> tuple[np.ndarray, np.ndarray]:
    return scores.noul_z >= 0, scores.choice_z.argmax(axis=-1)


def row(cells) -> str:
    return "| " + " | ".join(f"{c:.3f}" if isinstance(c, float) else str(c) for c in cells) + " |"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", type=Path)
    p.add_argument("--train", default="data/train-claude.jsonl")
    p.add_argument("--eval", default="data/eval-draft.jsonl")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    _, val_examples = split(load_examples(args.train), args.val_ratio, args.seed)
    eval_examples = load_examples(args.eval)

    evals: dict[str, Scores] = {}
    print("| モデル | 型 | 温度 | 正解率 | NLL | ECE | ECE (温度 1) | 検証データの正解率 |")
    print("|---|---|---|---|---|---|---|---|")
    for model_dir in args.models:
        val = load_scores(model_dir, "val", val_examples, args.device)
        evals[model_dir.name] = scores = load_scores(model_dir, "eval", eval_examples, args.device)
        config = InferenceConfig.load(model_dir)
        config.temperature_noul, config.temperature_choice = calibrate(val)
        config.save(model_dir)
        temperatures = {"noul": config.temperature_noul, "choice": config.temperature_choice}
        calibrated = metrics(scores, config.temperature_noul, config.temperature_choice)
        raw, val_metrics = metrics(scores), metrics(val)
        for kind in ("noul", "choice"):
            m = calibrated[kind]
            cells = [model_dir.name, kind, temperatures[kind], m["accuracy"], m["nll"], m["ece"]]
            print(row(cells + [raw[kind]["ece"], val_metrics[kind]["accuracy"]]))

    names = list(evals)
    first = evals[names[0]]
    print("\n判定が同じ割合 (評価セット。noul / choice)\n")
    print(row([""] + names))
    print("|---" * (len(names) + 1) + "|")
    for a in names:
        cells = [a]
        for b in names:
            (noul_a, choice_a), (noul_b, choice_b) = decisions(evals[a]), decisions(evals[b])
            cells.append(f"{np.mean(noul_a == noul_b):.3f} / {np.mean(choice_a == choice_b):.3f}")
        print(row(cells))

    print("\n質問ごと (評価セット)。noul は 真の割合 と、モデルごとの 正解率 / 真を当てた割合\n")
    print(row(["質問", "真の割合"] + names))
    print("|---" * (len(names) + 2) + "|")
    noul_names = np.array(first.noul_names)
    truth = first.noul_y >= 0.5
    for name in dict.fromkeys(first.noul_names):
        mask = noul_names == name
        cells = [name, float(truth[mask].mean())]
        for model in names:
            predicted = evals[model].noul_z[mask] >= 0
            hit = predicted == truth[mask]
            recall = hit[truth[mask]].mean() if truth[mask].any() else float("nan")
            cells.append(f"{hit.mean():.3f} / {recall:.3f}")
        print(row(cells))
    choice_names = np.array(first.choice_names)
    for name in dict.fromkeys(first.choice_names):
        mask = choice_names == name
        cells = [name, "(choice)"]
        for model in names:
            predicted = evals[model].choice_z[mask].argmax(axis=-1)
            cells.append(float(np.mean(predicted == first.choice_y[mask].argmax(axis=-1))))
        print(row(cells))


if __name__ == "__main__":
    main()
