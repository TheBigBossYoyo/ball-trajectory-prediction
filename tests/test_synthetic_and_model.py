import os

import numpy as np
import pytest

import synthetic as syn
from config import config
from model import TrajectoryPredictor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_velocity_feature_uses_forward_difference():
    traj = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 4.0]])
    out = syn._add_velocity_feature(traj)
    assert out[:, 2].tolist() == [2.0, 1.0, 1.0]


def test_falling_trajectory_stays_above_ground_and_ends_at_rest():
    traj = np.array(syn.generate_falling_trajectory(1.0, ball_radius=0.02))
    assert traj[0, 1] == 1.0
    assert np.all(traj[:, 1] >= 0.02 - 1e-12)
    assert traj[-1, 1] == pytest.approx(0.02)
    assert np.allclose(np.diff(traj[:, 0]), 0.02)


def test_sample_window_shapes_and_range():
    rng = np.random.RandomState(0)
    traj = syn.augment_trajectory(np.column_stack([np.arange(0, 1, 0.02),
                                                   np.linspace(0.6, 0.1, 50)]))
    for _ in range(20):
        x, y = syn.sample_window(traj, 50, 10, rng)
        assert x.shape == (50, 3) and y.shape == (10, 2)
        assert x.min() >= 0 and x.max() <= 1 and y.min() >= 0 and y.max() <= 1


def test_sample_window_past_end_is_a_resting_ball():
    rng = np.random.RandomState(0)
    t = np.arange(0, 0.4, 0.02) / config.TIME_MAX
    traj = np.column_stack([t, np.full(len(t), 0.05), np.zeros(len(t))])
    _, y = syn.sample_window(traj, 50, 10, rng)
    assert np.allclose(y[:, 1], 0.05)
    assert np.all(np.diff(y[:, 0]) > 0)


def test_augment_keeps_time_monotonic():
    np.random.seed(1)
    traj = np.column_stack([np.arange(0, 1, 0.02), np.linspace(0.6, 0.0, 50)])
    aug = syn.augment_trajectory(traj)
    assert np.all(np.diff(aug[:, 0]) > 0)


def test_real_trajectories_are_resampled_to_model_step(tmp_path):
    trajs = syn.load_real_trajectories([os.path.join(ROOT, "data.csv"),
                                        str(tmp_path / "nope.csv")])
    assert len(trajs) == 1
    assert np.allclose(np.diff(trajs[0][:, 0]), syn.MODEL_DT)


def test_generate_training_data_shapes_without_real_data():
    X, y = syn.generate_training_data(num_samples=10, include_real_data=False,
                                      windows_per_trajectory=2, seed=3)
    assert X.shape == (20, config.SEQUENCE_LENGTH, 3)
    assert y.shape == (20, config.PREDICTION_STEPS, 2)


def test_with_velocity_helper():
    pairs = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.2]])
    out = TrajectoryPredictor._with_velocity(pairs)
    assert out[:, 2] == pytest.approx([2.0, 0.0, 0.0])


class _Constant:
    """Stand-in for a Keras model: always predicts the same height."""
    def predict(self, x, verbose=0):
        assert x.shape == (1, config.SEQUENCE_LENGTH, 3)
        steps = config.PREDICTION_STEPS
        out = np.zeros((1, steps * 2))
        out[0, 1::2] = 0.25
        return out


def test_predict_full_extends_a_window_with_regular_time_steps():
    p = TrajectoryPredictor()
    p.model, p.is_trained = _Constant(), True
    seq = np.column_stack([np.arange(50) * 0.02 / 3.0, np.full(50, 0.3), np.zeros(50)])
    out = p.predict_full(seq, total_steps=80, max_iterations=10)
    assert out.shape == (80, 2)
    assert np.allclose(np.diff(out[:, 0]), 0.02 / 3.0)
    assert np.allclose(out[50:, 1], 0.25)


def test_predict_full_requires_trained_model():
    p = TrajectoryPredictor()
    with pytest.raises(ValueError):
        p.predict_full(np.zeros((50, 3)))
