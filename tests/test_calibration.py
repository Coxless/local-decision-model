import numpy as np
import pytest

from local_decision_model.calibration import expected_calibration_error, fit_temperature


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
