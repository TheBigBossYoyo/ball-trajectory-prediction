"""
Synthetic Data Generation Module - Training data for LSTM model.

This module generates synthetic trajectory data for training the LSTM
prediction model. It creates realistic falling ball trajectories with
bounces using physics simulation.

Example usage:
    from synthetic import generate_training_data

    X, y = generate_training_data(num_samples=800)
    # X shape: (800, 50, 2) - input sequences
    # y shape: (800, 10, 2) - prediction targets
"""
import os
import logging
import numpy as np
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config import config

logger = logging.getLogger(__name__)

# Use centralized config
RANDOM_SEED = config.RANDOM_SEED
GRAVITY = config.GRAVITY
AIR_DENSITY = config.AIR_DENSITY


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class TrajectoryParams:
    """Parameters for trajectory generation."""
    initial_speed: float
    launch_angle: float
    initial_height: float
    drag_coefficient: float
    ball_mass: float
    ball_radius: float


def generate_falling_trajectory(
    initial_height: float,
    ball_mass: float = 0.0592,
    ball_radius: float = 0.012,
    drag_coefficient: float = 0.47,
    dt: float = 0.02,
    include_bounces: bool = True,
    restitution: float = 0.75,
    max_bounces: int = 5
) -> List[Tuple[float, float]]:
    """
    Generate a falling ball trajectory with bounces.

    The trajectory is returned as (time, height) pairs, not (x, y) spatial coordinates.

    Args:
        initial_height: Starting height in meters
        ball_mass: Ball mass in kg
        ball_radius: Ball radius in meters
        drag_coefficient: Aerodynamic drag coefficient
        dt: Time step in seconds
        include_bounces: Whether to simulate bounces
        restitution: Bounce coefficient (0-1)
        max_bounces: Maximum number of bounces

    Returns:
        List of (time, height) tuples
    """
    area = np.pi * ball_radius ** 2
    drag_factor = 0.5 * AIR_DENSITY * drag_coefficient * area / ball_mass

    height = initial_height
    velocity = 0.0
    time = 0.0
    trajectory = [(time, height)]
    bounce_count = 0

    max_points = 500  # Prevent runaway trajectories
    # Use ball_radius as ground level (consistent with physics.py)
    ground_level = ball_radius

    while len(trajectory) < max_points:
        # Calculate drag acceleration for current velocity
        speed = abs(velocity)
        drag_acc = -np.sign(velocity) * drag_factor * velocity**2 if speed > 1e-10 else 0.0

        # Velocity Verlet integration (matches physics.py)
        # Half-step velocity
        vel_half = velocity + (-GRAVITY + drag_acc) * (dt / 2)
        # Full position update
        height += vel_half * dt
        # Recalculate acceleration at new position
        speed_half = abs(vel_half)
        new_drag_acc = -np.sign(vel_half) * drag_factor * vel_half**2 if speed_half > 1e-10 else 0.0
        # Complete velocity update
        velocity = vel_half + (-GRAVITY + new_drag_acc) * (dt / 2)
        time += dt

        # Ground collision (use ball_radius as ground level)
        if height <= ground_level:
            height = ground_level
            if include_bounces and abs(velocity) > 0.05 and bounce_count < max_bounces:
                # Apply restitution with decay (matches physics.py)
                effective_restitution = restitution * (0.95 ** bounce_count)
                velocity = -velocity * effective_restitution
                bounce_count += 1
            else:
                trajectory.append((time, ground_level))
                break

        trajectory.append((time, height))

    return trajectory


def generate_single_trajectory(params: TrajectoryParams, dt: float = 0.02) -> List[Tuple[float, float]]:
    """
    Generate a projectile trajectory (parabolic motion).

    Used for variety in training data. Returns (time, height) pairs.

    Args:
        params: Trajectory parameters
        dt: Time step

    Returns:
        List of (time, height) tuples
    """
    angle_rad = np.radians(params.launch_angle)
    vx = params.initial_speed * np.cos(angle_rad)
    vy = params.initial_speed * np.sin(angle_rad)
    x, y = 0.0, params.initial_height
    time = 0.0

    area = np.pi * params.ball_radius ** 2
    drag_factor = 0.5 * AIR_DENSITY * params.drag_coefficient * area / params.ball_mass

    trajectory = [(time, y)]

    for _ in range(1000):
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 1e-10:
            drag_ax = -drag_factor * speed * vx
            drag_ay = -drag_factor * speed * vy
        else:
            drag_ax = drag_ay = 0.0

        vx += drag_ax * dt
        vy += (-GRAVITY + drag_ay) * dt
        x += vx * dt
        y += vy * dt
        time += dt

        if y < 0:
            trajectory.append((time, 0.0))
            break

        trajectory.append((time, y))

    return trajectory


def _add_velocity_feature(trajectory: np.ndarray) -> np.ndarray:
    """
    Add velocity as a third feature to trajectory data.

    Computes velocity as (h[i+1] - h[i]) / (t[i+1] - t[i]) for each point.
    Uses forward difference with last point using backward difference.

    Args:
        trajectory: Array of shape (N, 2) with (time, height)

    Returns:
        Array of shape (N, 3) with (time, height, velocity)
    """
    n = len(trajectory)
    velocities = np.zeros(n)

    for i in range(n - 1):
        dt = trajectory[i + 1, 0] - trajectory[i, 0]
        if dt > 1e-8:
            velocities[i] = (trajectory[i + 1, 1] - trajectory[i, 1]) / dt

    # Last point uses backward difference
    if n > 1:
        dt = trajectory[-1, 0] - trajectory[-2, 0]
        if dt > 1e-8:
            velocities[-1] = (trajectory[-1, 1] - trajectory[-2, 1]) / dt

    return np.column_stack([trajectory, velocities])


def _normalize_trajectory_global(
    trajectory: np.ndarray,
    time_max: float = 3.0,
    height_max: float = 2.0
) -> np.ndarray:
    """
    Normalize trajectory using global bounds (consistent across all trajectories).

    This is better than per-trajectory normalization for model generalization.

    Args:
        trajectory: Raw trajectory array of shape (N, 2)
        time_max: Maximum expected time value
        height_max: Maximum expected height value

    Returns:
        Normalized trajectory with values in [0, 1]
    """
    traj_norm = trajectory.copy()

    # Normalize time
    traj_norm[:, 0] = traj_norm[:, 0] / time_max

    # Normalize height
    traj_norm[:, 1] = traj_norm[:, 1] / height_max

    # Clamp to [0, 1]
    traj_norm = np.clip(traj_norm, 0, 1)

    return traj_norm


def _pad_or_truncate(trajectory: np.ndarray, fixed_length: int) -> np.ndarray:
    """
    Pad or truncate trajectory to fixed length.

    Padding is added at the START with the first point to match
    inference behavior in model.py:prepare_sequence_for_prediction.

    Args:
        trajectory: Variable length trajectory
        fixed_length: Required length

    Returns:
        Fixed-length trajectory
    """
    if len(trajectory) >= fixed_length:
        return trajectory[:fixed_length]

    # Pad at START with first point (matches inference)
    padding = np.tile(trajectory[0:1], (fixed_length - len(trajectory), 1))
    return np.vstack([padding, trajectory])


def load_real_trajectories(data_files: List[str]) -> List[np.ndarray]:
    """
    Load real sensor data as training examples.

    Args:
        data_files: List of CSV file paths

    Returns:
        List of trajectory arrays
    """
    # Import here to avoid circular dependency
    from data_processing import process_sensor_csv

    trajectories = []

    for filepath in data_files:
        if not os.path.exists(filepath):
            logger.warning(f"Data file not found: {filepath}")
            continue

        try:
            time_s, _, height_m = process_sensor_csv(filepath)
            # Create trajectory array (time, height)
            traj = np.column_stack([time_s, height_m])
            trajectories.append(traj)
            logger.debug(f"Loaded real trajectory from {filepath}: {len(traj)} points")
        except Exception as e:
            logger.warning(f"Could not load {filepath}: {e}")

    return trajectories


def augment_trajectory(
    traj: np.ndarray,
    target_length: int,
    noise_scale: float = 0.01
) -> np.ndarray:
    """
    Augment a trajectory with noise and scaling for training variety.

    Args:
        traj: Original trajectory
        target_length: Required output length
        noise_scale: Standard deviation of noise to add

    Returns:
        Augmented trajectory
    """
    # Normalize using global bounds
    traj_norm = _normalize_trajectory_global(traj.copy())

    # Add small noise
    noise = np.random.normal(0, noise_scale, traj_norm.shape)
    traj_norm += noise

    # Random height scaling (simulate different drop heights)
    scale = np.random.uniform(0.8, 1.2)
    traj_norm[:, 1] *= scale

    # Clip and pad
    traj_norm = np.clip(traj_norm, 0, 1)
    # Add velocity feature before padding
    traj_norm = _add_velocity_feature(traj_norm)
    return _pad_or_truncate(traj_norm, target_length)


def generate_synthetic_dataset(
    num_trajectories: int = 1000,
    fixed_length: int = 200,
    seed: int = RANDOM_SEED
) -> List[np.ndarray]:
    """
    Generate synthetic trajectory dataset for training.

    Creates a mix of falling trajectories with bounces.

    Args:
        num_trajectories: Number of trajectories to generate
        fixed_length: Length of each trajectory
        seed: Random seed for reproducibility

    Returns:
        List of normalized trajectory arrays
    """
    set_seed(seed)
    trajectories = []

    for i in range(num_trajectories):
        # Mostly generate falling trajectories (more relevant to our use case)
        if random.random() < 0.8:
            # Falling trajectory with bounces - REALISTIC PARAMETERS
            traj = generate_falling_trajectory(
                initial_height=np.random.uniform(0.3, 2.0),
                ball_mass=np.random.uniform(0.003, 0.1),  # 3g to 100g
                ball_radius=np.random.uniform(0.01, 0.04),  # 10mm to 40mm
                drag_coefficient=np.random.uniform(0.4, 0.6),  # Realistic for spheres
                restitution=np.random.uniform(0.5, 0.9),
                max_bounces=np.random.randint(3, 8)
            )
        else:
            # Some projectile trajectories for variety
            params = TrajectoryParams(
                initial_speed=np.random.uniform(1, 5),  # Lower speeds for realistic bouncing
                launch_angle=np.random.uniform(0, 30),  # Lower angles
                initial_height=np.random.uniform(0.1, 1.0),
                drag_coefficient=np.random.uniform(0.4, 0.6),  # Realistic for spheres
                ball_mass=np.random.uniform(0.01, 0.1),
                ball_radius=np.random.uniform(0.01, 0.04)
            )
            traj = generate_single_trajectory(params)

        traj_array = np.array(traj)
        traj_array = _normalize_trajectory_global(traj_array)
        traj_array = _add_velocity_feature(traj_array)  # Add velocity as 3rd feature
        traj_array = _pad_or_truncate(traj_array, fixed_length)
        trajectories.append(traj_array)

    return trajectories


def generate_training_data(
    num_samples: int = None,
    sequence_length: int = None,
    prediction_steps: int = None,
    seed: int = None,
    include_real_data: bool = True,
    real_data_files: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for LSTM model.

    Creates input-output pairs for sequence-to-sequence prediction.
    Optionally includes augmented real sensor data.

    Args:
        num_samples: Number of training samples (default from config)
        sequence_length: Input sequence length (default from config)
        prediction_steps: Output prediction length (default from config)
        seed: Random seed (default from config)
        include_real_data: Whether to include augmented real sensor data
        real_data_files: List of real data CSV files

    Returns:
        Tuple of (X, y) arrays:
            X: Input sequences of shape (num_samples, sequence_length, 2)
            y: Target outputs of shape (num_samples, prediction_steps, 2)
    """
    # Use config defaults
    num_samples = num_samples or config.TRAINING_SAMPLES
    sequence_length = sequence_length or config.SEQUENCE_LENGTH
    prediction_steps = prediction_steps or config.PREDICTION_STEPS
    seed = seed or config.RANDOM_SEED

    set_seed(seed)
    total_length = sequence_length + prediction_steps

    # Calculate how many synthetic samples to generate
    real_augmentations = 0
    if include_real_data:
        if real_data_files is None:
            # Use default data files
            real_data_files = [f for f in config.DATA_FILES if os.path.exists(f)]
        real_augmentations = len(real_data_files) * 20  # 20 augmentations per real trajectory

    synthetic_count = max(0, num_samples - real_augmentations)

    # Generate synthetic trajectories
    trajectories = generate_synthetic_dataset(synthetic_count, total_length, seed)
    logger.info(f"Generated {len(trajectories)} synthetic trajectories")

    # Add augmented real data
    if include_real_data and real_data_files:
        real_trajs = load_real_trajectories(real_data_files)
        for traj in real_trajs:
            # Create multiple augmentations of each real trajectory
            for _ in range(20):
                augmented = augment_trajectory(traj, total_length)
                trajectories.append(augmented)
        logger.info(f"Added {len(real_trajs) * 20} augmented real trajectories")

    # Shuffle trajectories
    random.shuffle(trajectories)

    # Create X, y pairs
    X = np.array([t[:sequence_length] for t in trajectories])
    y = np.array([t[sequence_length:total_length] for t in trajectories])

    logger.info(f"Training data: X shape {X.shape}, y shape {y.shape}")

    return X, y
