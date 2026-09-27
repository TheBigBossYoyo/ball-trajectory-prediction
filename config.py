"""
Configuration module for Trajectory Prediction System.

Consolidates all constants and supports environment variables for deployment flexibility.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """
    Application configuration with environment variable support.

    All settings can be overridden via environment variables.
    """

    # Data files
    DATA_FILES: List[str] = field(default_factory=lambda: ["data.csv", "data2.csv", "data3.csv"])
    DEFAULT_DATA_FILE: str = "data.csv"
    MODEL_FILE: str = "model.keras"
    NORM_PARAMS_FILE: str = "norm_params.json"

    # Physics constants
    GRAVITY: float = 9.81
    AIR_DENSITY: float = 1.25

    # Model parameters
    RANDOM_SEED: int = 42
    SEQUENCE_LENGTH: int = 50
    PREDICTION_STEPS: int = 10
    TRAINING_SAMPLES: int = 800
    TRAINING_EPOCHS: int = 30

    # Feature configuration
    NUM_FEATURES: int = 3  # time, height, velocity
    OUTPUT_FEATURES: int = 1  # Only predict height (time is deterministic)

    # Sensor configuration
    SENSOR_HEIGHT_CM: float = 70.0
    MAX_SENSOR_READING: float = 66.0

    # Normalization bounds
    TIME_MAX: float = 3.0
    HEIGHT_MAX: float = 2.0

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = False

    # Simulation defaults
    DEFAULT_DROP_HEIGHT_CM: float = 67.0
    DEFAULT_RESTITUTION: float = 0.85
    DEFAULT_INITIAL_VELOCITY: float = 0.0
    MAX_SIMULATION_TIME: float = 3.0
    SIMULATION_DT: float = 0.002
    MAX_BOUNCES: int = 8

    def __post_init__(self):
        """Override settings from environment variables."""
        self.HOST = os.getenv("HOST", self.HOST)
        self.PORT = int(os.getenv("PORT", str(self.PORT)))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        self.RANDOM_SEED = int(os.getenv("RANDOM_SEED", str(self.RANDOM_SEED)))
        self.TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", str(self.TRAINING_EPOCHS)))
        self.TRAINING_SAMPLES = int(os.getenv("TRAINING_SAMPLES", str(self.TRAINING_SAMPLES)))

        # Model file can be overridden
        self.MODEL_FILE = os.getenv("MODEL_FILE", self.MODEL_FILE)


# Global config instance
config = Config()
