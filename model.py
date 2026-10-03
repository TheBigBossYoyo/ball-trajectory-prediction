"""
LSTM Model Module - Neural network for trajectory prediction.

This module provides an LSTM-based neural network for predicting
ball trajectories based on initial sequence observations.

Example usage:
    from model import TrajectoryPredictor

    predictor = TrajectoryPredictor()
    predictor.build()
    predictor.train(X, y, epochs=30)
    prediction = predictor.predict(initial_sequence)
"""
import os
import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("TensorFlow not available. LSTM model will be disabled.")
    TF_AVAILABLE = False
    # Mock classes to prevent NameError
    tf = None
    keras = None
    EarlyStopping = None
    ReduceLROnPlateau = None

from config import config

logger = logging.getLogger(__name__)

# Set seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
if TF_AVAILABLE:
    tf.random.set_seed(config.RANDOM_SEED)


@dataclass
class NormalizationParams:
    """Saved normalization parameters for consistent inference."""
    time_min: float = 0.0
    time_max: float = 3.0
    height_min: float = 0.0
    height_max: float = 2.0


def build_lstm_model(
    sequence_length: int = 50,
    num_features: int = 3,
    prediction_steps: int = 10,
    use_bidirectional: bool = True
):
    """
    Build enhanced LSTM model architecture.

    Architecture: Bidirectional LSTM(128) -> Dropout -> LSTM(64) -> Dropout
                  -> Dense(64) -> Dense(output)

    Args:
        sequence_length: Length of input sequences
        num_features: Number of features per timestep (time, height, velocity)
        prediction_steps: Number of future steps to predict
        use_bidirectional: Whether to use bidirectional first layer

    Returns:
        Compiled Keras model or None if TF unavailable
    """
    if not TF_AVAILABLE:
        return None

    from tensorflow.keras.layers import Bidirectional

    layers = [Input(shape=(sequence_length, num_features))]

    # First LSTM layer (optionally bidirectional)
    if use_bidirectional:
        layers.append(Bidirectional(LSTM(128, return_sequences=True)))
    else:
        layers.append(LSTM(128, return_sequences=True))

    layers.extend([
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        # Output height predictions for each step (time is deterministic)
        Dense(prediction_steps * 2, activation='linear')  # Keep 2 for (time, height) pairs
    ])

    model = Sequential(layers)

    # Add gradient clipping to prevent exploding gradients
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model


def train_model(model, X_train, y_train, epochs=100, batch_size=32, verbose=1):
    """
    Train the LSTM model with early stopping.

    Args:
        model: Keras model to train
        X_train: Training input sequences
        y_train: Training target sequences
        epochs: Maximum training epochs
        batch_size: Training batch size
        verbose: Verbosity level

    Returns:
        Training history object
    """
    if not TF_AVAILABLE or model is None:
        return None

    # Flatten y if needed
    if len(y_train.shape) == 3:
        y_train = y_train.reshape(y_train.shape[0], -1)

    # Shuffle before splitting to avoid ordering bias
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    split_idx = int(0.8 * len(X_train))
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )


def predict_trajectory(model, input_sequence, prediction_steps=10):
    """
    Predict future trajectory from input sequence.

    Args:
        model: Trained Keras model
        input_sequence: Input sequence of shape (seq_len, 2)
        prediction_steps: Number of steps to predict

    Returns:
        Predicted points of shape (prediction_steps, 2)
    """
    if len(input_sequence.shape) == 2:
        input_sequence = np.expand_dims(input_sequence, axis=0)
    prediction = model.predict(input_sequence, verbose=0)
    return prediction.reshape(-1, 2)


class TrajectoryPredictor:
    """
    High-level interface for trajectory prediction.

    Handles model building, training, saving, loading, and prediction
    with proper normalization.
    """

    def __init__(
        self,
        model_path: str = None,
        norm_path: str = None
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model file
            norm_path: Path to normalization parameters file
        """
        self.model_path = model_path or config.MODEL_FILE
        self.norm_path = norm_path or config.NORM_PARAMS_FILE
        self.model = None # Type: Optional[keras.Model]
        self.sequence_length = config.SEQUENCE_LENGTH
        self.prediction_steps = config.PREDICTION_STEPS
        self.is_trained = False
        self.norm_params = NormalizationParams()

    def load(self) -> bool:
        """
        Load model and normalization parameters with validation.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not TF_AVAILABLE:
            return False

        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}")
            return False

        try:
            self.model = load_model(self.model_path)

            # Validate architecture
            expected_input = (None, self.sequence_length, config.NUM_FEATURES)
            expected_output = (None, self.prediction_steps * 2)

            if self.model.input_shape != expected_input:
                raise ValueError(
                    f"Model input shape {self.model.input_shape} != expected {expected_input}"
                )
            if self.model.output_shape != expected_output:
                raise ValueError(
                    f"Model output shape {self.model.output_shape} != expected {expected_output}"
                )

            # Load normalization parameters
            self._load_norm_params()

            self.is_trained = True
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.is_trained = False
            return False

    def build(self):
        """Build a new model architecture."""
        self.model = build_lstm_model(
            self.sequence_length, config.NUM_FEATURES, self.prediction_steps
        )
        self.is_trained = False
        logger.info("Built new model")

    def train(self, X, y, epochs=None, verbose=1):
        """
        Train the model on provided data.

        Args:
            X: Training input sequences
            y: Training target sequences
            epochs: Training epochs (uses config default if None)
            verbose: Verbosity level

        Returns:
            Training history object
        """
        if self.model is None:
            self.build()

        epochs = epochs or config.TRAINING_EPOCHS
        history = train_model(self.model, X, y, epochs=epochs, verbose=verbose)
        self.is_trained = True

        return history

    def save(self):
        """Save model and normalization parameters."""
        if self.model:
            self.model.save(self.model_path)
            self._save_norm_params()
            logger.info(f"Model saved to {self.model_path}")

    def predict(self, input_sequence: np.ndarray) -> np.ndarray:
        """
        Predict next steps from input sequence.

        Args:
            input_sequence: Input of shape (seq_len, 2)

        Returns:
            Predicted points of shape (prediction_steps, 2)

        Raises:
            ValueError: If model not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        return predict_trajectory(self.model, input_sequence, self.prediction_steps)

    def predict_full(
        self,
        initial_sequence: np.ndarray,
        total_steps: int = 200,
        max_iterations: int = 100
    ) -> np.ndarray:
        """
        Predict full trajectory using autoregressive prediction.

        Args:
            initial_sequence: Starting sequence of shape (seq_len, 2)
            total_steps: Total number of points to generate
            max_iterations: Maximum prediction iterations (prevents infinite loop)

        Returns:
            Full trajectory of shape (total_steps, 2): normalised (time, height),
            starting with the input window

        Raises:
            ValueError: If model not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Ensure initial sequence is correct length
        if len(initial_sequence) < self.sequence_length:
            # Pad from start with first value
            padding = np.tile(
                initial_sequence[0:1],
                (self.sequence_length - len(initial_sequence), 1)
            )
            initial_sequence = np.vstack([padding, initial_sequence])

        # Rows are (time, height, velocity), normalised. The network predicts
        # (time, height) pairs; time is deterministic, so each predicted point
        # gets the next time on the regular grid, and velocity is recomputed
        # from heights the same way it is for training data.
        dt_norm = 0.02 / (self.norm_params.time_max - self.norm_params.time_min)
        rows = np.array(initial_sequence[-self.sequence_length:], dtype=float)
        times = list(rows[:, 0])
        heights = list(rows[:, 1])
        iterations = 0

        while len(heights) < total_steps and iterations < max_iterations:
            pred = self.predict(rows)

            if pred is None or len(pred) == 0:
                logger.warning("Empty prediction, stopping")
                break

            for point in pred:
                times.append(times[-1] + dt_norm)
                heights.append(float(np.clip(point[1], 0.0, 1.0)))
                if len(heights) >= total_steps:
                    break

            rows = self._with_velocity(
                np.column_stack([times[-self.sequence_length:],
                                 heights[-self.sequence_length:]])
            )
            iterations += 1

        return np.column_stack([times, heights])[:total_steps]

    @staticmethod
    def _with_velocity(pairs: np.ndarray) -> np.ndarray:
        """Append velocity (forward difference, last row backward) to (time, height) rows."""
        n = len(pairs)
        vel = np.zeros(n)
        dt = np.diff(pairs[:, 0])
        dh = np.diff(pairs[:, 1])
        ok = dt > 1e-8
        vel[:-1][ok] = dh[ok] / dt[ok]
        if n > 1 and ok[-1]:
            vel[-1] = dh[-1] / dt[-1]
        return np.column_stack([pairs, vel])

    def prepare_initial_sequence(
        self,
        height: float,
        velocity: float = 0.0
    ) -> np.ndarray:
        """
        Create well-conditioned initial sequence for prediction.

        Generates physically realistic initial points to avoid startup glitches.

        Args:
            height: Initial drop height in meters
            velocity: Initial vertical velocity in m/s (negative = down)

        Returns:
            Initial sequence of shape (sequence_length, num_features)
        """
        dt = 0.02
        g = config.GRAVITY

        points = []
        t = 0.0
        y = height
        v = velocity

        for i in range(self.sequence_length):
            # Normalize time and height
            t_norm = t / self.norm_params.time_max
            h_norm = y / self.norm_params.height_max
            # Compute normalized velocity (change in normalized height per normalized time)
            v_norm = v / (self.norm_params.height_max / self.norm_params.time_max)
            points.append([t_norm, h_norm, v_norm])

            # Simple physics step
            v -= g * dt
            y += v * dt
            y = max(0, y)
            t += dt

        return np.array(points)

    def normalize(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Normalize trajectory using saved parameters.

        Args:
            trajectory: Raw trajectory of shape (N, 2)

        Returns:
            Normalized trajectory of shape (N, 2)
        """
        t_norm = (trajectory[:, 0] - self.norm_params.time_min) / \
                 (self.norm_params.time_max - self.norm_params.time_min + 1e-8)
        h_norm = (trajectory[:, 1] - self.norm_params.height_min) / \
                 (self.norm_params.height_max - self.norm_params.height_min + 1e-8)
        return np.column_stack([t_norm, h_norm])

    def denormalize_height(self, normalized_height: np.ndarray) -> np.ndarray:
        """
        Denormalize height values back to physical units.

        Uses stored normalization parameters for consistency.

        Args:
            normalized_height: Normalized height values (0-1 range)

        Returns:
            Height values in meters
        """
        height_range = self.norm_params.height_max - self.norm_params.height_min
        return normalized_height * height_range + self.norm_params.height_min

    def denormalize_time(self, normalized_time: np.ndarray) -> np.ndarray:
        """
        Denormalize time values back to seconds.

        Args:
            normalized_time: Normalized time values (0-1 range)

        Returns:
            Time values in seconds
        """
        time_range = self.norm_params.time_max - self.norm_params.time_min
        return normalized_time * time_range + self.norm_params.time_min

    def denormalize(self, normalized_trajectory: np.ndarray) -> np.ndarray:
        """
        Denormalize full trajectory (time, height) back to physical units.

        Args:
            normalized_trajectory: Normalized trajectory of shape (N, 2)

        Returns:
            Denormalized trajectory with physical units
        """
        result = normalized_trajectory.copy()
        result[:, 0] = self.denormalize_time(normalized_trajectory[:, 0])
        result[:, 1] = self.denormalize_height(normalized_trajectory[:, 1])
        return result

    def _save_norm_params(self):
        """Save normalization parameters to JSON."""
        with open(self.norm_path, 'w') as f:
            json.dump(asdict(self.norm_params), f, indent=2)
        logger.debug(f"Saved normalization params to {self.norm_path}")

    def _load_norm_params(self):
        """Load normalization parameters from JSON."""
        if os.path.exists(self.norm_path):
            try:
                with open(self.norm_path, 'r') as f:
                    params = json.load(f)
                    self.norm_params = NormalizationParams(**params)
                logger.debug(f"Loaded normalization params from {self.norm_path}")
            except Exception as e:
                logger.warning(f"Could not load norm params: {e}")


def prepare_sequence_for_prediction(trajectory: np.ndarray, sequence_length: int = 50) -> np.ndarray:
    """
    Prepare trajectory for model prediction.

    Pads or truncates to required sequence length.

    Args:
        trajectory: Input trajectory of shape (N, num_features)
        sequence_length: Required sequence length

    Returns:
        Prepared sequence of shape (sequence_length, num_features)
    """
    if len(trajectory) >= sequence_length:
        return trajectory[-sequence_length:]

    padding = np.tile(trajectory[0:1], (sequence_length - len(trajectory), 1))
    return np.vstack([padding, trajectory])
