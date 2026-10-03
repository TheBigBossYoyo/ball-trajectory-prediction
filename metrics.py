"""
Error metrics for comparing a predicted trajectory with a measured one.

All heights are in metres and all times in seconds. These are plain NumPy
functions so they can be tested without TensorFlow or matplotlib.
"""
from typing import Optional, Tuple

import numpy as np


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """Root-mean-square error."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("pred and truth must have the same non-zero size")
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or pred.size == 0:
        raise ValueError("pred and truth must have the same non-zero size")
    return float(np.mean(np.abs(pred - truth)))


def first_crossing_time(
    time: np.ndarray, height: np.ndarray, threshold: float
) -> Optional[float]:
    """
    First time the height drops to `threshold` or below (linear interpolation
    between samples). Used as the "time of first impact" because the sensor
    cannot see the exact instant of contact. Returns None if it never does.
    """
    time, height = np.asarray(time, float), np.asarray(height, float)
    below = np.where(height <= threshold)[0]
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return float(time[0])
    h0, h1 = height[i - 1], height[i]
    frac = (h0 - threshold) / (h0 - h1) if h0 != h1 else 0.0
    return float(time[i - 1] + frac * (time[i] - time[i - 1]))


def window_apex(
    time: np.ndarray, height: np.ndarray, t_start: float, t_end: float
) -> Optional[Tuple[float, float]]:
    """
    (time, height) of the highest sample with t_start <= time <= t_end, or None
    if no sample falls in that window. Used for the first rebound peak.
    """
    time, height = np.asarray(time, float), np.asarray(height, float)
    sel = np.where((time >= t_start) & (time <= t_end))[0]
    if len(sel) == 0:
        return None
    i = sel[np.argmax(height[sel])]
    return float(time[i]), float(height[i])
