"""Physics engine checks against analytic solutions and basic physical laws."""
import numpy as np
import pytest

from physics import Ball, BallShape, BALL_PRESETS, simulate_trajectory_3d
from config import config

G = config.GRAVITY


def run(ball, h, **kw):
    r = simulate_trajectory_3d(ball, initial_height=h, **kw)
    return np.array(r["time"]), np.array(r["y"])


def peaks(y):
    return [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1]]


def terminal_speed(ball):
    return np.sqrt(G / ball.drag_factor)


def test_free_fall_without_drag_matches_analytic_solution():
    ball = BALL_PRESETS["steel"]
    h0 = 2.0
    t, y = run(ball, h0, dt=0.001, include_drag=False, max_time=0.5)
    analytic = h0 - 0.5 * G * t ** 2
    fall = analytic > ball.radius  # before the first impact
    assert np.max(np.abs(y - analytic)[fall]) < 1e-9


def test_free_fall_impact_time_without_drag():
    ball = BALL_PRESETS["steel"]
    h0 = 1.0
    dt = 0.0005
    t, y = run(ball, h0, dt=dt, include_drag=False, max_time=1.0)
    expected = np.sqrt(2 * (h0 - ball.radius) / G)
    first_hit = t[np.argmax(y <= ball.radius + 1e-12)]
    assert abs(first_hit - expected) <= 2 * dt


def test_free_fall_with_drag_matches_analytic_solution():
    # Closed form for quadratic drag from rest: y = h0 - (vt^2/g) ln cosh(g t / vt)
    ball = BALL_PRESETS["rubber"]
    vt = terminal_speed(ball)
    h0 = 5.0
    t, y = run(ball, h0, dt=0.001, max_time=0.8)
    analytic = h0 - vt ** 2 / G * np.log(np.cosh(G * t / vt))
    fall = analytic > ball.radius
    assert np.max(np.abs(y - analytic)[fall]) < 1e-5


def test_drag_makes_the_fall_slower():
    ball = BALL_PRESETS["ping_pong"]
    t, y_drag = run(ball, 3.0, include_drag=True, max_time=0.6)
    _, y_free = run(ball, 3.0, include_drag=False, max_time=0.6)
    n = min(len(y_drag), len(y_free))
    assert np.all(y_drag[1:n] >= y_free[1:n] - 1e-12)
    assert y_drag[n - 1] > y_free[n - 1]


def test_speed_approaches_terminal_velocity():
    ball = BALL_PRESETS["ping_pong"]
    vt = terminal_speed(ball)
    dt = 0.002
    t, y = run(ball, 200.0, dt=dt, max_time=6.0)
    v = (y[-1] - y[-2]) / dt
    assert abs(-v - vt) / vt < 0.01


def test_energy_decreases_with_drag_and_is_conserved_without():
    ball = BALL_PRESETS["rubber"]
    dt = 0.001
    for drag in (True, False):
        t, y = run(ball, 4.0, dt=dt, include_drag=drag, max_time=0.7)
        fall = y > ball.radius + 0.05
        t, y = t[fall], y[fall]
        v = np.gradient(y, dt)  # central differences, second-order accurate
        energy = ball.mass * G * y + 0.5 * ball.mass * v ** 2
        energy = energy[2:-2]   # skip the one-sided differences at the ends
        if drag:
            assert np.all(np.diff(energy) < 0)
            assert energy[-1] < energy[0] - 1e-4
        else:
            assert np.max(np.abs(energy - energy[0])) / energy[0] < 1e-6


def test_restitution_reduces_rebound_height_by_e_squared():
    ball = BALL_PRESETS["steel"]
    e = 0.8
    h0 = 1.0
    t, y = run(ball, h0, dt=0.0005, include_drag=False, restitution=e,
               restitution_decay=1.0, max_time=2.5)
    above_floor = y - ball.radius
    top = [above_floor[i] for i in peaks(y)]
    drop = h0 - ball.radius
    assert top[0] == pytest.approx(drop * e ** 2, rel=0.01)
    assert top[1] == pytest.approx(drop * e ** 4, rel=0.02)


def test_restitution_decay_applies_per_bounce():
    ball = BALL_PRESETS["steel"]
    e, k = 0.9, 0.8
    t, y = run(ball, 1.0, dt=0.0005, include_drag=False, restitution=e,
               restitution_decay=k, max_time=2.5)
    top = [(y - ball.radius)[i] for i in peaks(y)]
    # bounce 0 uses e, bounce 1 uses e*k
    assert top[1] / top[0] == pytest.approx((e * k) ** 2, rel=0.03)


def test_perfectly_elastic_ball_returns_to_drop_height_without_drag():
    ball = BALL_PRESETS["steel"]
    t, y = run(ball, 1.0, dt=0.0005, include_drag=False, restitution=1.0,
               restitution_decay=1.0, max_time=1.2)
    assert max(y[peaks(y)[0]:]) == pytest.approx(1.0, abs=0.01)


def test_timestep_convergence_is_second_order():
    ball = BALL_PRESETS["rubber"]
    vt = terminal_speed(ball)
    h0 = 3.0

    def max_error(dt):
        t, y = run(ball, h0, dt=dt, max_time=0.5)
        analytic = h0 - vt ** 2 / G * np.log(np.cosh(G * t / vt))
        return np.max(np.abs(y - analytic))

    errors = [max_error(dt) for dt in (0.008, 0.004, 0.002)]
    assert errors[0] > errors[1] > errors[2]
    # halving dt should cut the error by about 4 (second order); allow slack
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


def test_first_impact_time_converges_with_smaller_timestep():
    ball = BALL_PRESETS["steel"]
    exact = np.sqrt(2 * (1.0 - ball.radius) / G)
    errs = []
    for dt in (0.01, 0.0025, 0.0005):
        t, y = run(ball, 1.0, dt=dt, include_drag=False, max_time=1.0)
        errs.append(abs(t[np.argmax(y <= ball.radius + 1e-12)] - exact))
    assert errs[-1] < errs[0]
    assert errs[-1] <= 0.0005 * 2


def test_ball_never_goes_below_ground_and_comes_to_rest():
    ball = BALL_PRESETS["tennis"]
    t, y = run(ball, 1.5, max_time=10.0)
    assert np.all(y >= ball.radius - 1e-12)
    assert y[-1] == pytest.approx(ball.radius, abs=1e-3)
    assert t[-1] < 10.0  # stopped early


def test_zero_gravity_is_respected():
    # gravity=0 used to be silently replaced by the default 9.81
    ball = BALL_PRESETS["steel"]
    t, y = run(ball, 1.0, gravity=0.0, include_drag=False, max_time=0.5)
    assert np.allclose(y, 1.0)


def test_initial_velocity_is_used():
    ball = BALL_PRESETS["steel"]
    t, y = run(ball, 1.0, initial_velocity=(0.0, -2.0, 0.0), dt=0.001,
               include_drag=False, max_time=0.2)
    analytic = 1.0 - 2.0 * t - 0.5 * G * t ** 2
    assert np.max(np.abs(y - analytic)) < 1e-9


def test_heavier_ball_feels_less_drag():
    base = BALL_PRESETS["ping_pong"]
    light = Ball("light", 0.001, base.radius, BallShape.SPHERE)
    heavy = Ball("heavy", 1.0, base.radius, BallShape.SPHERE)
    _, y_light = run(light, 5.0, max_time=0.8)
    _, y_heavy = run(heavy, 5.0, max_time=0.8)
    n = min(len(y_light), len(y_heavy))
    assert y_light[n - 1] > y_heavy[n - 1]


def test_drag_factor_formula_and_cube_area():
    ball = BALL_PRESETS["rubber"]
    expected = 0.5 * config.AIR_DENSITY * 0.47 * np.pi * ball.radius ** 2 / ball.mass
    assert ball.drag_factor == pytest.approx(expected)
    cube = Ball("c", 1.0, 0.1, BallShape.CUBE)
    assert cube.cross_sectional_area == pytest.approx(0.2 ** 2)


def test_horizontal_velocity_is_damped_at_each_bounce():
    ball = BALL_PRESETS["steel"]
    r = simulate_trajectory_3d(ball, 0.5, initial_velocity=(1.0, 0.0, 0.0),
                               include_drag=False, max_time=1.5)
    x = np.array(r["x"])
    assert x[-1] > 0.2
    assert np.all(np.diff(x) >= -1e-12)
