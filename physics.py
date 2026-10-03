"""
Physics Module - 3D Equations of motion with air drag.

This module provides physics simulation for ball trajectories including:
- Newtonian mechanics with gravitational acceleration
- Air drag based on ball properties
- Bounce physics with coefficient of restitution
- Support for multiple ball types (steel, rubber, etc.)

Example usage:
    from physics import BALL_PRESETS, simulate_trajectory_3d

    ball = BALL_PRESETS['steel']
    result = simulate_trajectory_3d(ball, initial_height=0.67)

    print(f"Fall time: {result['time'][-1]:.2f}s")
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from enum import Enum

from config import config

# Use centralized config
GRAVITY = config.GRAVITY
AIR_DENSITY = config.AIR_DENSITY


class BallShape(Enum):
    """Ball shape types with their drag coefficients."""
    SPHERE = 0.47
    HOLLOW_SPHERE = 0.50
    CUBE = 1.05


@dataclass
class Ball:
    """
    Ball properties for physics simulation.

    Attributes:
        name: Display name of the ball
        mass: Mass in kilograms
        radius: Radius in meters
        shape: Shape type (affects drag coefficient)
        material: Material name for display
        color: Hex color for rendering
        restitution: Coefficient of restitution (bounciness)
        is_rigid: Whether the ball deforms on impact (visual only)
        associated_dataset: Optional filename of matching sensor data
    """
    name: str
    mass: float
    radius: float
    shape: BallShape
    material: str = "generic"
    color: str = "#FF6B35"
    restitution: float = 0.85
    is_rigid: bool = True
    associated_dataset: str = None

    @property
    def drag_coefficient(self) -> float:
        """Get drag coefficient from shape type."""
        return self.shape.value

    @property
    def cross_sectional_area(self) -> float:
        """Calculate cross-sectional area for drag calculation."""
        if self.shape == BallShape.CUBE:
            return (self.radius * 2) ** 2
        return np.pi * self.radius ** 2

    @property
    def drag_factor(self) -> float:
        """
        Pre-computed drag factor for efficiency.

        Returns F_drag / v^2 = 0.5 * rho * Cd * A / m
        """
        return (0.5 * AIR_DENSITY * self.drag_coefficient *
                self.cross_sectional_area / self.mass)


# Pre-defined ball presets with realistic properties
BALL_PRESETS: Dict[str, Ball] = {
    "steel": Ball(
        name="Steel Ball",
        mass=0.0592,        # 59.2g
        radius=0.012,       # 12mm radius
        shape=BallShape.SPHERE,
        material="steel",
        color="#7C7C7C",
        restitution=0.6,    # Reduced to 0.6 for faster energy loss
        is_rigid=True,
        associated_dataset=None
    ),
    "rubber": Ball(
        name="Rubber Ball",
        mass=0.059,         # 59g - matches experimental data
        radius=0.03,        # 30mm radius
        shape=BallShape.SPHERE,
        material="rubber",
        color="#E63946",
        restitution=0.85,   # Slightly reduced for realism
        is_rigid=False,
        associated_dataset="data.csv"
    ),
    "ping_pong": Ball(
        name="Ping-Pong Ball",
        mass=0.0027,        # 2.7g
        radius=0.02,        # 20mm radius
        shape=BallShape.HOLLOW_SPHERE,
        material="celluloid",
        color="#FFFFFF",
        restitution=0.89,   # Very bouncy
        is_rigid=True
    ),
    "tennis": Ball(
        name="Tennis Ball",
        mass=0.057,         # 57g
        radius=0.033,       # 33mm radius
        shape=BallShape.SPHERE,
        material="felt",
        color="#CCFF00",
        restitution=0.75,
        is_rigid=False
    ),
    "custom": Ball(
        name="Custom Ball",
        mass=0.05,          # 50g
        radius=0.025,       # 25mm radius
        shape=BallShape.SPHERE,
        material="custom",
        color="#9B59B6",
        restitution=0.8,
        is_rigid=True
    )
}


def simulate_trajectory_3d(
    ball: Ball,
    initial_height: float,
    initial_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    initial_position: Tuple[float, float] = (0.0, 0.0),
    dt: float = None,
    max_time: float = None,
    include_drag: bool = True,
    restitution: float = None,
    gravity: float = None,
    restitution_decay: float = 0.95
) -> Dict[str, Any]:
    """
    Simulate 3D ball trajectory with physics.

    Uses velocity Verlet integration for accurate physics simulation
    including gravitational acceleration, air drag, and elastic bouncing.

    Args:
        ball: Ball object with mass, radius, and drag properties
        initial_height: Starting height in meters
        initial_velocity: (vx, vy, vz) in m/s, negative vy = downward
        initial_position: (x, z) starting position in meters
        dt: Time step in seconds (default from config)
        max_time: Maximum simulation time in seconds (default from config)
        include_drag: Whether to include air resistance
        restitution: Coefficient of restitution for bounces (default from config)
        gravity: Gravitational acceleration in m/s^2 (default from config)
        restitution_decay: Factor applied to the restitution once per bounce, so
            bounce n uses restitution * restitution_decay**n (1.0 = constant)

    Returns:
        Dictionary with keys:
            - time: List of time points in seconds
            - x, y, z: Lists of positions in meters
            - ballRadius: Ball radius in meters
            - ballColor: Hex color string
    """
    # Use config defaults
    dt = dt if dt is not None else config.SIMULATION_DT
    max_time = max_time if max_time is not None else config.MAX_SIMULATION_TIME
    restitution = restitution if restitution is not None else ball.restitution
    gravity = gravity if gravity is not None else config.GRAVITY
    if dt <= 0:
        raise ValueError("dt must be positive")

    positions = []
    times = []

    pos = np.array([initial_position[0], initial_height, initial_position[1]])
    vel = np.array(list(initial_velocity))

    bounce_count = 0
    max_bounces = config.MAX_BOUNCES
    min_vel = 0.01

    t = 0.0

    while t < max_time:
        positions.append(pos.copy())
        times.append(t)

        # Gravitational acceleration
        acc = np.array([0.0, -gravity, 0.0])

        # Air drag (opposes velocity)
        if include_drag:
            speed = np.linalg.norm(vel)
            if speed > 1e-10:
                drag_mag = ball.drag_factor * speed ** 2
                acc -= (vel / speed) * drag_mag

        # Velocity Verlet step. Drag depends on velocity, so the acceleration
        # at the end of the step needs a velocity estimate: use a forward-Euler
        # prediction of the end-of-step velocity, then average the two
        # accelerations (second-order accurate).
        pos = pos + vel * dt + 0.5 * acc * dt ** 2
        vel_pred = vel + acc * dt

        acc_new = np.array([0.0, -gravity, 0.0])
        if include_drag:
            speed = np.linalg.norm(vel_pred)
            if speed > 1e-10:
                drag_mag = ball.drag_factor * speed ** 2
                acc_new -= (vel_pred / speed) * drag_mag

        vel = vel + 0.5 * (acc + acc_new) * dt

        # Ground collision detection and response
        if pos[1] <= ball.radius:
            pos[1] = ball.radius

            impact_velocity = abs(vel[1])

            if impact_velocity > min_vel and bounce_count < max_bounces:
                # Apply restitution with a per-bounce decay for extra energy loss
                effective_restitution = restitution * (restitution_decay ** bounce_count)
                vel[1] = -vel[1] * effective_restitution

                # Friction reduces horizontal velocity (increase friction slightly)
                vel[0] *= 0.95
                vel[2] *= 0.95

                bounce_count += 1
            else:
                # Ball has stopped
                vel = np.array([0.0, 0.0, 0.0])
                pos[1] = ball.radius

        t += dt

        # Early termination if ball is at rest
        if pos[1] <= ball.radius + 0.001 and np.linalg.norm(vel) < min_vel:
            positions.append(pos.copy())
            times.append(t)
            break

    pos_array = np.array(positions)

    return {
        'time': times,
        'x': pos_array[:, 0].tolist(),
        'y': pos_array[:, 1].tolist(),
        'z': pos_array[:, 2].tolist(),
        'ballRadius': ball.radius,
        'ballColor': ball.color,
        'isRigid': ball.is_rigid
    }


def calculate_drag_factor(
    mass: float,
    radius: float,
    drag_coefficient: float = 0.47
) -> float:
    """
    Calculate drag factor for custom ball parameters.

    Args:
        mass: Ball mass in kg
        radius: Ball radius in meters
        drag_coefficient: Drag coefficient (default 0.47 for sphere)

    Returns:
        Drag factor (F_drag / v^2)
    """
    area = np.pi * radius ** 2
    return 0.5 * AIR_DENSITY * drag_coefficient * area / mass
