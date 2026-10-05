import numpy as np
import pytest

from local_decision_model.calibration import (
    choice_metrics,
    expected_calibration_error,
    fit_temperature,
    fit_temperature_choice,
)


def test_fit_temperature_recovers_true_temperature():
    rng = np.random.default_rng(0)
    true_t = 3.0
    z = rng.normal(0, 6, size=20000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-z / true_t))).astype(float)
    assert fit_temperature(z, y) == pytest.approx(true_t, rel=0.1)


def test_ece_perfect_and_bad():
    y = np.array([1, 0, 1, 0], dtype=float)
    assert expected_calibration_error(y, y) == pytest.approx(0.0)
    assert expected_calibration_error(1 - y, y) == pytest.approx(1.0)


def test_fit_temperature_choice_recovers_true_temperature():
    rng = np.random.default_rng(0)
    true_t = 2.5
    z = rng.normal(0, 4, size=(20000, 4))
    z[::2, 3] = -np.inf  # 半分は 3 択 (足りない列は -inf)
    p = np.exp(z / true_t)
    p /= p.sum(axis=1, keepdims=True)
    picked = (p.cumsum(axis=1) > rng.random((len(z), 1))).argmax(axis=1)
    y = np.eye(4)[picked]
    assert fit_temperature_choice(z, y) == pytest.approx(true_t, rel=0.1)


def test_choice_metrics_ignore_padding():
    z = np.array([[2.0, 0.0, -np.inf], [0.0, 1.0, 3.0]])
    y = np.array([[0.2, 0.8, 0.0], [0.0, 0.1, 0.9]])
    m = choice_metrics(z, y)
    assert m["accuracy"] == pytest.approx(0.5)
    p0 = np.exp(2.0) / (np.exp(2.0) + 1)
    p1 = np.exp([0.0, 1.0, 3.0]) / np.exp([0.0, 1.0, 3.0]).sum()
    nll = -(0.2 * np.log(p0) + 0.8 * np.log(1 - p0) + 0.1 * np.log(p1[1]) + 0.9 * np.log(p1[2]))
    assert m["nll"] == pytest.approx(nll / 2)
