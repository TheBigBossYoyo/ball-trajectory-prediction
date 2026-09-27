import unittest
import numpy as np

class TestSampling(unittest.TestCase):
    def test_downsample_upsample_logic(self):
        # Simulate high-freq physics data (500Hz)
        dt_physics = 0.002
        duration = 1.0
        n_points = int(duration / dt_physics)
        physics_times = np.linspace(0, duration, n_points)
        # Simple sine wave
        physics_values = np.sin(2 * np.pi * 1.0 * physics_times)

        # Downsample to 50Hz (factor of 10)
        step_ratio = 10
        input_points = []

        # Logic from app.py
        for i in range(0, n_points, step_ratio):
            input_points.append(physics_values[i])

        self.assertEqual(len(input_points), n_points // 10)

        # Simulate "Prediction" (just identity for this test)
        # Pred times would be 0, 0.02, 0.04...
        pred_times = physics_times[::step_ratio]
        pred_values = np.array(input_points)

        # Upsample back to 500Hz
        upsampled_values = np.interp(
            physics_times,
            pred_times,
            pred_values
        )

        # Verify reconstruction is close
        # (It won't be exact due to linear interpolation of a curve, but should be close)
        correlation = np.corrcoef(physics_values, upsampled_values)[0, 1]
        self.assertGreater(correlation, 0.99)

        # Check alignment at known points
        self.assertAlmostEqual(upsampled_values[0], physics_values[0])
        self.assertAlmostEqual(upsampled_values[100], physics_values[100], places=1)

if __name__ == '__main__':
    unittest.main()
