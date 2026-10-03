import numpy as np
import pytest

from metrics import rmse, mae, first_crossing_time, window_apex


def test_rmse_and_mae_known_values():
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.0, 4.0, 3.0])
    assert rmse(pred, truth) == pytest.approx(np.sqrt(4 / 3))
    assert mae(pred, truth) == pytest.approx(2 / 3)


def test_zero_error_for_identical_curves():
    a = np.linspace(0, 1, 20)
    assert rmse(a, a) == 0.0
    assert mae(a, a) == 0.0


def test_rmse_is_at_least_mae():
    rng = np.random.RandomState(0)
    a, b = rng.rand(50), rng.rand(50)
    assert rmse(a, b) >= mae(a, b)


def test_constant_offset_gives_equal_rmse_and_mae():
    a = np.linspace(0, 1, 10)
    assert rmse(a + 0.1, a) == pytest.approx(0.1)
    assert mae(a + 0.1, a) == pytest.approx(0.1)


def test_shape_mismatch_and_empty_input_raise():
    with pytest.raises(ValueError):
        rmse(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        mae(np.array([]), np.array([]))


def test_first_crossing_time_interpolates_between_samples():
    t = np.array([0.0, 1.0, 2.0])
    h = np.array([1.0, 0.6, 0.2])
    assert first_crossing_time(t, h, 0.5) == pytest.approx(1.25)


def test_first_crossing_time_edge_cases():
    t = np.array([0.0, 1.0])
    assert first_crossing_time(t, np.array([1.0, 0.9]), 0.5) is None
    assert first_crossing_time(t, np.array([0.1, 0.9]), 0.5) == 0.0


def test_window_apex_finds_peak_inside_window_only():
    t = np.linspace(0, 2, 201)
    h = np.abs(np.sin(np.pi * t))   # peaks at 0.5 and 1.5
    h[150] = 5.0                    # outside the window below
    time, height = window_apex(t, h, 0.2, 1.0)
    assert time == pytest.approx(0.5, abs=0.011)
    assert height == pytest.approx(1.0, abs=1e-3)
    assert window_apex(t, h, 3.0, 4.0) is None
