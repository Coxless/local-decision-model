"""確率の較正 (temperature scaling) と評価指標。

jev の売りは「較正された確率」なので、学習後に検証データで温度 T を求めて
P(真) = sigmoid(z / T) が実際の正解率と合うようにする。
"""

from __future__ import annotations

import numpy as np


def _nll(z: np.ndarray, y: np.ndarray, temperature: float) -> float:
    # 数値的に安定な binary cross entropy (y は 0〜1 のソフトラベル可)
    s = z / temperature
    return float(np.mean(np.logaddexp(0.0, s) - y * s))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """検証データの判断ロジット z とラベル y から NLL 最小の温度を求める。"""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # log T 上で粗く探索してから黄金分割で詰める (1 変数なので十分)
    grid = np.linspace(np.log(0.05), np.log(20.0), 200)
    best = grid[int(np.argmin([_nll(z, y, np.exp(g)) for g in grid]))]
    lo, hi = best - 0.05, best + 0.05
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        a = hi - phi * (hi - lo)
        b = lo + phi * (hi - lo)
        if _nll(z, y, np.exp(a)) < _nll(z, y, np.exp(b)):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """ECE: 確率を bins 個に区切り、各区間の平均確率と正解率の差を件数で重み付けした平均。"""
    p = np.asarray(p, dtype=np.float64)
    y = (np.asarray(y, dtype=np.float64) >= 0.5).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def binary_metrics(z: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    p = 1.0 / (1.0 + np.exp(-np.asarray(z) / temperature))
    hard = np.asarray(y) >= 0.5
    return {
        "accuracy": float(np.mean((p >= 0.5) == hard)),
        "nll": _nll(np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64), temperature),
        "ece": expected_calibration_error(p, y),
    }
