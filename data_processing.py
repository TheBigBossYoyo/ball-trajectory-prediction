"""
Data Processing Module - Sensor data handling and transformation.

This module provides functions for loading, cleaning, and processing
sensor data from CSV files for trajectory analysis.

Example usage:
    from data_processing import process_sensor_csv

    time_s, x_m, height_m = process_sensor_csv("data.csv")
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from scipy import interpolate
from scipy.signal import savgol_filter

from config import config

logger = logging.getLogger(__name__)

# Use config values
SENSOR_HEIGHT_CM = config.SENSOR_HEIGHT_CM
MAX_SENSOR_READING = config.MAX_SENSOR_READING


def load_sensor_data(filepath: str) -> pd.DataFrame:
    """
    Load sensor data from CSV file with validation.

    Args:
        filepath: Path to CSV file with semicolon separator

    Returns:
        DataFrame with 'time_ms' and 'distance_cm' columns

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sensor data file not found: {filepath}")

    df = pd.read_csv(filepath, sep=';')

    # Validate structure
    if len(df.columns) != 2:
        raise ValueError(f"Expected 2 columns, got {len(df.columns)} in {filepath}")

    df.columns = ['time_ms', 'distance_cm']

    # Validate numeric types
    if not pd.api.types.is_numeric_dtype(df['time_ms']):
        raise ValueError(f"time_ms column must be numeric in {filepath}")
    if not pd.api.types.is_numeric_dtype(df['distance_cm']):
        raise ValueError(f"distance_cm column must be numeric in {filepath}")

    # Validate non-empty
    if len(df) == 0:
        raise ValueError(f"Sensor data file is empty: {filepath}")

    logger.debug(f"Loaded {len(df)} rows from {filepath}")
    return df


def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sensor data by handling invalid readings.

    Replaces readings >= MAX_SENSOR_READING with interpolated values,
    except for trailing invalid readings which are clamped to ground level.

    Args:
        df: Raw sensor DataFrame

    Returns:
        Cleaned DataFrame with interpolated missing values
    """
    df_clean = df.copy()

    # Find indices of invalid readings
    invalid_mask = df_clean['distance_cm'] >= MAX_SENSOR_READING

    # Count invalid readings for logging
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logger.debug(f"Replacing {invalid_count} invalid sensor readings")

    # Find where the ball has settled (many consecutive invalid readings at the end)
    # This indicates the ball is at rest on the ground (too close for sensor)
    invalid_indices = np.where(invalid_mask)[0]

    if len(invalid_indices) > 0:
        # Check for trailing sequence of invalid readings
        last_valid_idx = -1
        for i in range(len(df_clean) - 1, -1, -1):
            if not invalid_mask.iloc[i]:
                last_valid_idx = i
                break

        # If there are trailing invalid readings (ball at rest on ground)
        if last_valid_idx < len(df_clean) - 1:
            # Count consecutive invalid readings at the end
            trailing_count = len(df_clean) - 1 - last_valid_idx

            # If more than 5 consecutive invalid readings at end, ball is at rest
            # Set these to sensor_height (meaning ball is at ground level = 0 height)
            if trailing_count > 5:
                # Use SENSOR_HEIGHT_CM as distance (ball at floor = height 0)
                df_clean.loc[last_valid_idx + 1:, 'distance_cm'] = SENSOR_HEIGHT_CM
                logger.debug(f"Clamped {trailing_count} trailing readings to ground level")

    # Now handle remaining invalid readings in the middle (interpolate)
    # Re-check which are still invalid after trailing fix
    still_invalid = df_clean['distance_cm'] >= MAX_SENSOR_READING
    df_clean.loc[still_invalid, 'distance_cm'] = np.nan
    df_clean['distance_cm'] = df_clean['distance_cm'].interpolate(method='linear')
    df_clean['distance_cm'] = df_clean['distance_cm'].ffill().bfill()

    return df_clean


def invert_sensor_to_height(distance_cm: np.ndarray) -> np.ndarray:
    """
    Convert sensor distance readings to ball height.

    The sensor measures distance from above, so height = sensor_height - distance.

    Args:
        distance_cm: Array of distance readings in cm

    Returns:
        Array of height values in cm (clamped to >= 0)
    """
    height = SENSOR_HEIGHT_CM - distance_cm
    return np.maximum(height, 0)


def smooth_trajectory(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter to trajectory data.

    Args:
        data: Raw trajectory data array
        window_size: Filter window size (must be odd)

    Returns:
        Smoothed trajectory data
    """
    if len(data) < window_size:
        return data

    if window_size % 2 == 0:
        window_size += 1

    # Ensure window doesn't exceed data length
    if window_size > len(data):
        window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_size < 3:
            return data

    try:
        return savgol_filter(data, window_size, polyorder=2)
    except ValueError:
        return data


def resample_trajectory(
    time_ms: np.ndarray,
    height_cm: np.ndarray,
    target_dt_ms: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample trajectory to uniform time steps using safe interpolation.

    Uses linear interpolation with bounded extrapolation for safety.

    Args:
        time_ms: Original time points in milliseconds
        height_cm: Height values in centimeters
        target_dt_ms: Target time step in milliseconds

    Returns:
        Tuple of (resampled_time_ms, resampled_height_cm)
    """
    if len(time_ms) < 2:
        return time_ms, height_cm

    new_time = np.arange(time_ms[0], time_ms[-1], target_dt_ms)

    # Use linear interpolation with clamped bounds (safer than cubic extrapolation)
    f = interpolate.interp1d(
        time_ms, height_cm,
        kind='linear',
        bounds_error=False,
        fill_value=(height_cm[0], height_cm[-1])
    )

    new_height = np.maximum(f(new_time), 0)
    return new_time, new_height


def process_sensor_csv(
    filepath: str,
    smooth: bool = True,
    resample: bool = True,
    extract_first_only: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete sensor data processing pipeline.

    Loads CSV, cleans data, converts to height, optionally smooths and resamples.

    Args:
        filepath: Path to sensor CSV file
        smooth: Apply Savitzky-Golay smoothing
        resample: Resample to uniform time steps
        extract_first_only: Reserved for future multi-trajectory files

    Returns:
        Tuple of (time_seconds, x_meters, height_meters)
    """
    df = load_sensor_data(filepath)
    df = clean_sensor_data(df)

    time_ms = df['time_ms'].values
    distance_cm = df['distance_cm'].values
    height_cm = invert_sensor_to_height(distance_cm)

    if smooth:
        height_cm = smooth_trajectory(height_cm)

    if resample:
        time_ms, height_cm = resample_trajectory(time_ms, height_cm)

    # Convert to SI units
    time_s = time_ms / 1000.0
    height_m = height_cm / 100.0
    x_m = np.zeros_like(height_m)

    return time_s, x_m, height_m


def extend_trajectory_with_physics(
    time_s: np.ndarray,
    height_m: np.ndarray,
    target_duration: float = 2.0,
    restitution: float = 0.75
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extend trajectory using simple physics simulation.

    Continues the trajectory using ballistic motion if it ends before target duration.

    Args:
        time_s: Original time array in seconds
        height_m: Original height array in meters
        target_duration: Target total duration in seconds
        restitution: Bounce coefficient (0-1)

    Returns:
        Tuple of (extended_time, extended_height)
    """
    if len(time_s) < 2 or time_s[-1] >= target_duration:
        return time_s, height_m

    dt = time_s[-1] - time_s[-2] if len(time_s) > 1 else 0.01
    final_vel = (height_m[-1] - height_m[-2]) / dt if dt > 0 else 0.0

    g = config.GRAVITY
    t = time_s[-1]
    h = height_m[-1]
    v = final_vel

    ext_time = list(time_s)
    ext_height = list(height_m)

    while t < target_duration:
        t += 0.01
        v -= g * 0.01
        h += v * 0.01

        if h <= 0:
            h = 0
            v = -v * restitution if abs(v) > 0.1 else 0

        ext_time.append(t)
        ext_height.append(h)

    return np.array(ext_time), np.array(ext_height)
