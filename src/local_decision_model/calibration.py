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


def _log_softmax(z: np.ndarray, temperature: float) -> np.ndarray:
    s = z / temperature
    return s - np.logaddexp.reduce(s, axis=-1, keepdims=True)


def _nll_choice(z: np.ndarray, y: np.ndarray, temperature: float) -> float:
    # y は選択肢ごとの確率。z が -inf の列 (選択肢の少ない質問の埋め草) は y が 0 なので数えない
    log_p = np.where(y > 0, _log_softmax(z, temperature), 0.0)
    return float(-np.mean(np.sum(y * log_p, axis=-1)))


def _fit(nll) -> float:
    # log T 上で粗く探索してから黄金分割で詰める (1 変数なので十分)
    grid = np.linspace(np.log(0.05), np.log(20.0), 200)
    best = grid[int(np.argmin([nll(np.exp(g)) for g in grid]))]
    lo, hi = best - 0.05, best + 0.05
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        a = hi - phi * (hi - lo)
        b = lo + phi * (hi - lo)
        if nll(np.exp(a)) < nll(np.exp(b)):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """検証データの判断ロジット z とラベル y から NLL 最小の温度を求める (noul)。"""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _fit(lambda t: _nll(z, y, t))


def fit_temperature_choice(z: np.ndarray, y: np.ndarray) -> float:
    """choice の温度。z と y は [質問の数, 選択肢の最大数] で、足りない列は z が -inf、y が 0。"""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _fit(lambda t: _nll_choice(z, y, t))


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


def choice_metrics(z: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    """choice の指標。正解は y が最大の選択肢。ECE は最上位の答えの確率で測る。"""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.exp(_log_softmax(z, temperature))
    correct = p.argmax(axis=-1) == y.argmax(axis=-1)
    return {
        "accuracy": float(np.mean(correct)),
        "nll": _nll_choice(z, y, temperature),
        "ece": expected_calibration_error(p.max(axis=-1), correct),
    }
