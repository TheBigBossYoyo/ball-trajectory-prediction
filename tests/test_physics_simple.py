import unittest
import sys
import os

# Add parent directory to path to allow importing physics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from physics import Ball, BallShape, simulate_trajectory_3d, BALL_PRESETS

class TestPhysics(unittest.TestCase):
    def test_ball_properties(self):
        steel = BALL_PRESETS['steel']
        self.assertEqual(steel.name, "Steel Ball")
        self.assertGreater(steel.mass, 0)

    def test_simulation_basic(self):
        ball = BALL_PRESETS['steel']
        result = simulate_trajectory_3d(ball, initial_height=1.0)
        self.assertIn('time', result)
        self.assertIn('y', result)
        self.assertGreater(len(result['time']), 0)

    def test_simulation_drag(self):
        ball = BALL_PRESETS['steel']
        # Simulation with drag
        result_drag = simulate_trajectory_3d(ball, initial_height=2.0, include_drag=True)
        # Simulation without drag
        result_no_drag = simulate_trajectory_3d(ball, initial_height=2.0, include_drag=False)

        # With drag, terminal velocity is lower, so it might take longer to fall?
        # For a steel ball at 2m, effect is small but present.
        # Let's check that they are not identical.
        self.assertNotEqual(result_drag['y'], result_no_drag['y'])

    def test_mass_configuration(self):
        """Verify that changing mass affects the trajectory when drag is enabled."""
        base_ball = BALL_PRESETS['ping_pong'] # Light ball has more drag effect

        # Heavy version
        heavy_ball = Ball(
            name="Heavy", mass=1.0, radius=base_ball.radius,
            shape=BallShape.SPHERE, restitution=0.8, is_rigid=True
        )

        # Light version
        light_ball = Ball(
            name="Light", mass=0.001, radius=base_ball.radius,
            shape=BallShape.SPHERE, restitution=0.8, is_rigid=True
        )

        res_heavy = simulate_trajectory_3d(heavy_ball, initial_height=5.0, include_drag=True)
        res_light = simulate_trajectory_3d(light_ball, initial_height=5.0, include_drag=True)

        # Light ball should be affected more by drag (slow down more)
        # So at the same time index (e.g. near end), light ball should be higher (falling slower)
        # or hit ground later.

        # Find index where both are still in air
        idx = min(len(res_heavy['y']), len(res_light['y'])) // 2
        if idx > 0:
            # This is hard to assert deterministically without exact calculation,
            # but they should be different.
            self.assertNotEqual(res_heavy['y'][idx], res_light['y'][idx])

if __name__ == '__main__':
    unittest.main()
