import os

import numpy as np
import pandas as pd
import pytest

import data_processing as dp
from config import config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def write_csv(tmp_path, rows, header="time_after_drop_ms;distance_cm"):
    p = tmp_path / "d.csv"
    p.write_text(header + "\n" + "\n".join(f"{a};{b}" for a, b in rows) + "\n")
    return str(p)


def test_load_sensor_data_reads_semicolon_csv(tmp_path):
    df = dp.load_sensor_data(write_csv(tmp_path, [(0, 3), (10, 4)]))
    assert list(df.columns) == ["time_ms", "distance_cm"]
    assert len(df) == 2


def test_load_sensor_data_rejects_bad_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        dp.load_sensor_data(str(tmp_path / "missing.csv"))
    three_cols = tmp_path / "c.csv"
    three_cols.write_text("a;b;c\n1;2;3\n")
    with pytest.raises(ValueError):
        dp.load_sensor_data(str(three_cols))
    empty = tmp_path / "e.csv"
    empty.write_text("a;b\n")
    with pytest.raises(ValueError):
        dp.load_sensor_data(str(empty))


def test_clean_interpolates_isolated_glitches():
    df = pd.DataFrame({"time_ms": [0, 10, 20, 30, 40],
                       "distance_cm": [10.0, 20.0, 70.0, 40.0, 50.0]})
    out = dp.clean_sensor_data(df)
    assert out["distance_cm"].tolist() == pytest.approx([10, 20, 30, 40, 50])


def test_clean_clamps_long_trailing_run_to_ground():
    # Regression test: the clamped value (70) is above the invalid threshold and
    # used to be turned back into NaN and filled with the last good reading.
    dist = [10.0, 20.0, 30.0] + [68.0] * 8
    df = pd.DataFrame({"time_ms": np.arange(len(dist)) * 10, "distance_cm": dist})
    out = dp.clean_sensor_data(df)
    assert out["distance_cm"].iloc[-1] == dp.SENSOR_HEIGHT_CM
    assert dp.invert_sensor_to_height(out["distance_cm"].values)[-1] == 0


def test_clean_does_not_clamp_short_trailing_run():
    dist = [10.0, 20.0, 30.0, 68.0, 68.0]
    df = pd.DataFrame({"time_ms": np.arange(5) * 10, "distance_cm": dist})
    out = dp.clean_sensor_data(df)
    assert out["distance_cm"].tolist() == pytest.approx([10, 20, 30, 30, 30])


def test_invert_sensor_to_height_clamps_at_zero():
    h = dp.invert_sensor_to_height(np.array([3.0, 70.0, 75.0]))
    assert h.tolist() == [67.0, 0.0, 0.0]


def test_smooth_keeps_quadratic_and_handles_short_input():
    t = np.arange(20, dtype=float)
    y = 0.5 * t ** 2
    assert dp.smooth_trajectory(y) == pytest.approx(y)  # SG order 2 is exact on a parabola
    short = np.array([1.0, 2.0])
    assert dp.smooth_trajectory(short).tolist() == [1.0, 2.0]


def test_resample_gives_uniform_grid_and_linear_values():
    t = np.array([0.0, 8.0, 17.0, 31.0, 40.0])
    h = 2 * t
    nt, nh = dp.resample_trajectory(t, h, target_dt_ms=10.0)
    assert np.allclose(np.diff(nt), 10.0)
    assert nh == pytest.approx(2 * nt)


def test_process_sensor_csv_units_and_grid():
    t, x, h = dp.process_sensor_csv(os.path.join(ROOT, "data.csv"))
    assert np.allclose(np.diff(t), 0.01)
    assert h[0] == pytest.approx(0.67, abs=0.02)   # sensor is 70 cm up, ball starts 3 cm below it
    assert np.all(h >= 0) and np.all(h <= 0.7)
    assert np.all(x == 0)


def test_process_all_shipped_datasets_load():
    for name in config.DATA_FILES:
        t, _, h = dp.process_sensor_csv(os.path.join(ROOT, name))
        assert len(t) == len(h) > 50


def test_valid_reading_mask_flags_interpolated_samples():
    raw_t = np.array([0, 10, 20, 30, 40, 50, 60])
    raw_d = np.array([5, 6, 68, 68, 68, 9, 10])
    grid = np.arange(0, 61, 10) / 1000.0
    mask = dp.valid_reading_mask(raw_t, raw_d, grid, tolerance_ms=5)
    assert mask.tolist() == [True, True, False, False, False, True, True]


def test_valid_reading_mask_all_invalid():
    mask = dp.valid_reading_mask([0, 10], [70, 70], np.array([0.0, 0.01]))
    assert not mask.any()


def test_extend_trajectory_reaches_target_and_stays_nonnegative():
    t = np.arange(0, 0.5, 0.01)
    h = np.maximum(0.5 - 0.5 * 9.81 * t ** 2, 0.0) + 0.1
    et, eh = dp.extend_trajectory_with_physics(t, h, target_duration=1.5)
    assert et[-1] >= 1.5
    assert np.all(eh >= 0)
    et2, eh2 = dp.extend_trajectory_with_physics(t, h, target_duration=0.2)
    assert len(et2) == len(t)
