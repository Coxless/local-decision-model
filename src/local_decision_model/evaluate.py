"""学習データの行に対する判断ロジットの収集、温度の較正、評価指標。

noul と choice を別々に扱う。温度は検証データ (学習データから分けたもの) で求め、
評価セットにはその温度をそのまま使う。
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .calibration import binary_metrics, choice_metrics, fit_temperature, fit_temperature_choice
from .decide import Decider
from .distill import Example
from .schema import Choice, parse_questions


@dataclass
class Scores:
    """質問ごとの判断ロジット z とラベル y。noul と choice で分けて持つ。"""

    noul_names: list[str]
    noul_z: np.ndarray  # [noul の数]
    noul_y: np.ndarray
    choice_names: list[str]
    choice_z: np.ndarray  # [choice の数, 選択肢の最大数]。足りない列は -inf
    choice_y: np.ndarray  # 足りない列は 0

    def save(self, path: str | Path) -> None:
        """npz に書く (あとで較正や比較をやり直すときに、推論し直さなくて済む)。"""
        np.savez(path, **asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> Scores:
        with np.load(path) as data:
            return cls(
                data["noul_names"].tolist(),
                data["noul_z"],
                data["noul_y"],
                data["choice_names"].tolist(),
                data["choice_z"],
                data["choice_y"],
            )


def score(decider: Decider, examples: list[Example], log_every: int = 100) -> Scores:
    noul_names, noul_z, noul_y = [], [], []
    choice_names, choice_z, choice_y = [], [], []
    for count, example in enumerate(examples, 1):
        questions = parse_questions(example.questions)
        z = decider.logits(example.state, questions)
        for name, q in questions.items():
            if isinstance(q, Choice):
                choice_names.append(name)
                choice_z.append(np.asarray(z[name], dtype=np.float64))
                choice_y.append([example.labels[name][o] for o in q.options])
            else:
                noul_names.append(name)
                noul_z.append(float(z[name][0]))
                noul_y.append(example.labels[name])
        if count % log_every == 0:
            print(f"{count} / {len(examples)} 行", file=sys.stderr)

    width = max((len(z) for z in choice_z), default=0)
    padded_z = np.full((len(choice_z), width), -np.inf)
    padded_y = np.zeros((len(choice_z), width))
    for i, (z, y) in enumerate(zip(choice_z, choice_y, strict=True)):
        padded_z[i, : len(z)] = z
        padded_y[i, : len(y)] = y
    return Scores(noul_names, np.array(noul_z), np.array(noul_y), choice_names, padded_z, padded_y)


def calibrate(scores: Scores) -> tuple[float, float]:
    """(noul の温度, choice の温度)。その型の質問がなければ 1。

    ラベルは 0 / 1 に丸めてから合わせる (noul は 0.5 以上が真、choice は確率が最大の選択肢)。
    確率のまま合わせると、温度は教師の確率のあいまいさに合い、正解率とは合わなくなる。
    """
    noul = choice = 1.0
    if scores.noul_names:
        noul = fit_temperature(scores.noul_z, scores.noul_y >= 0.5)
    if scores.choice_names:
        top = np.eye(scores.choice_y.shape[1])[scores.choice_y.argmax(axis=-1)]
        choice = fit_temperature_choice(scores.choice_z, top)
    return noul, choice


def metrics(scores: Scores, temperature_noul: float = 1.0, temperature_choice: float = 1.0) -> dict:
    """型ごとの正解率・NLL・ECE と、質問名ごとの正解率。"""
    report: dict = {}
    by_question: dict[str, list[bool]] = {}
    if scores.noul_names:
        report["noul"] = binary_metrics(scores.noul_z, scores.noul_y, temperature_noul)
        report["noul"]["count"] = len(scores.noul_names)
        correct = (scores.noul_z >= 0) == (scores.noul_y >= 0.5)
        for name, c in zip(scores.noul_names, correct, strict=True):
            by_question.setdefault(name, []).append(bool(c))
    if scores.choice_names:
        report["choice"] = choice_metrics(scores.choice_z, scores.choice_y, temperature_choice)
        report["choice"]["count"] = len(scores.choice_names)
        correct = scores.choice_z.argmax(axis=-1) == scores.choice_y.argmax(axis=-1)
        for name, c in zip(scores.choice_names, correct, strict=True):
            by_question.setdefault(name, []).append(bool(c))
    report["accuracy_by_question"] = {n: float(np.mean(c)) for n, c in by_question.items()}
    return report
