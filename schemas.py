"""
Request/Response schemas with Pydantic validation.

Provides type-safe API interfaces with automatic validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

from config import config


class SimulationRequest(BaseModel):
    """
    Validated simulation request parameters.

    All parameters have sensible defaults and validation bounds.
    """
    dropHeight: float = Field(
        default=config.DEFAULT_DROP_HEIGHT_CM,
        ge=1,
        le=500,
        description="Drop height in centimeters"
    )
    ballType: str = Field(
        default="steel",
        description="Ball type preset"
    )
    enablePhysics: bool = Field(
        default=True,
        description="Enable physics model"
    )
    enableSensor: bool = Field(
        default=True,
        description="Enable sensor data model"
    )
    enableLstm: bool = Field(
        default=True,
        description="Enable LSTM prediction model"
    )
    gravity: float = Field(
        default=config.GRAVITY,
        ge=0.1,
        le=50,
        description="Gravitational acceleration in m/s^2"
    )
    restitution: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
        description="Coefficient of restitution for bounces"
    )
    mass: Optional[float] = Field(
        default=None,
        ge=0.001,
        le=1.0,
        description="Ball mass in kg"
    )
    initialVelocity: float = Field(
        default=config.DEFAULT_INITIAL_VELOCITY,
        ge=-10,
        le=10,
        description="Initial vertical velocity in m/s"
    )
    dragEnabled: bool = Field(
        default=True,
        description="Enable air drag simulation"
    )
    lstmBlend: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="LSTM blend factor (0=physics only, 1=LSTM only)"
    )
    sensorDataFile: str = Field(
        default=config.DEFAULT_DATA_FILE,
        description="Sensor data file to use"
    )

    @field_validator('ballType')
    @classmethod
    def validate_ball_type(cls, v: str) -> str:
        valid_types = ['steel', 'rubber', 'ping_pong', 'tennis', 'custom']
        if v not in valid_types:
            raise ValueError(f'ballType must be one of {valid_types}')
        return v

    @field_validator('sensorDataFile')
    @classmethod
    def validate_data_file(cls, v: str) -> str:
        if v not in config.DATA_FILES:
            raise ValueError(f'sensorDataFile must be one of {config.DATA_FILES}')
        return v


class TrajectoryData(BaseModel):
    """Trajectory output data for a single model."""
    time: List[float]
    x: List[float]
    y: List[float]
    z: List[float]
    ballRadius: float
    ballColor: str
    label: str


class ErrorDetail(BaseModel):
    """Error detail structure."""
    code: str
    message: str
    details: Optional[dict] = None


class SimulationResponse(BaseModel):
    """Full simulation response."""
    physics: Optional[TrajectoryData] = None
    sensor: Optional[TrajectoryData] = None
    lstm: Optional[TrajectoryData] = None
    error: Optional[ErrorDetail] = None
