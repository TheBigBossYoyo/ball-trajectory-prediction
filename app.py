"""
Flask Application - Trajectory Prediction System API.

Provides REST API endpoints for ball trajectory simulation using
physics models, sensor data, and LSTM neural network predictions.

Usage:
    python app.py
    # Server starts at http://localhost:5000
"""
import os
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import config
from schemas import SimulationRequest, SimulationResponse, TrajectoryData, ErrorDetail
from physics import Ball, BALL_PRESETS, simulate_trajectory_3d
from data_processing import process_sensor_csv, extend_trajectory_with_physics
from synthetic import generate_training_data, set_seed
from model import TrajectoryPredictor, TF_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SensorDataset:
    """Container for processed sensor data."""
    time: List[float]
    height: List[float]
    name: str
    point_count: int


@dataclass
class AppState:
    """
    Thread-safe application state container.

    Replaces global variables with a proper state management pattern.
    Uses threading.Lock for safe concurrent access.
    """
    predictor: Optional[TrajectoryPredictor] = None
    sensor_datasets: Dict[str, SensorDataset] = field(default_factory=dict)
    current_dataset: str = "data.csv"
    model_trained: bool = False
    initialization_complete: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_predictor(self, predictor: TrajectoryPredictor, trained: bool = False):
        """Thread-safe predictor update."""
        with self._lock:
            self.predictor = predictor
            self.model_trained = trained

    def add_sensor_dataset(self, name: str, time: List[float], height: List[float]):
        """Thread-safe sensor data update."""
        with self._lock:
            self.sensor_datasets[name] = SensorDataset(
                time=time,
                height=height,
                name=name,
                point_count=len(time)
            )

    def get_sensor_data(self, name: str = None) -> Optional[SensorDataset]:
        """Get sensor data by name or current default."""
        with self._lock:
            key = name or self.current_dataset
            return self.sensor_datasets.get(key)

    def mark_initialized(self):
        """Mark initialization as complete."""
        with self._lock:
            self.initialization_complete = True


# Global application state
app_state = AppState()

# Flask app
app = Flask(__name__)
CORS(app)


def load_sensor_dataset(filepath: str, name: str) -> bool:
    """
    Load a single sensor dataset.

    Args:
        filepath: Path to sensor CSV file
        name: Dataset identifier name

    Returns:
        True if loaded successfully, False otherwise
    """
    if not os.path.exists(filepath):
        logger.warning(f"Sensor data file not found: {filepath}")
        return False

    try:
        time_s, x_m, y_m = process_sensor_csv(filepath)

        # Extend short trajectories
        if time_s[-1] < 2.0:
            time_s, y_m = extend_trajectory_with_physics(
                time_s, y_m,
                target_duration=2.5,
                restitution=config.DEFAULT_RESTITUTION
            )

        app_state.add_sensor_dataset(name, time_s.tolist(), y_m.tolist())
        logger.info(f"Loaded {name}: {len(time_s)} data points")
        return True

    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return False


def initialize_app():
    """
    Initialize application state including sensor data and LSTM model.

    Loads all available sensor datasets and trains/loads the LSTM model.
    """
    logger.info("=" * 50)
    logger.info("Trajectory Prediction System")
    logger.info("=" * 50)

    set_seed(config.RANDOM_SEED)

    # Load all sensor datasets
    logger.info("[1/3] Loading sensor datasets...")
    base_dir = os.path.dirname(__file__)
    loaded_count = 0

    for data_file in config.DATA_FILES:
        # Try both relative and absolute paths
        paths = [
            data_file,
            os.path.join(base_dir, data_file)
        ]

        for path in paths:
            if load_sensor_dataset(path, data_file):
                loaded_count += 1
                break

    logger.info(f"  Loaded {loaded_count}/{len(config.DATA_FILES)} datasets")

    # Create synthetic fallback if no data loaded
    if loaded_count == 0:
        logger.warning("  Creating synthetic fallback data...")
        t = np.linspace(0, 2.0, 200)
        h = 0.67 * np.exp(-2 * t) * np.abs(np.cos(5 * t))
        app_state.add_sensor_dataset("synthetic", t.tolist(), h.tolist())
        app_state.current_dataset = "synthetic"
    else:
        app_state.current_dataset = config.DEFAULT_DATA_FILE

    # Initialize LSTM model
    logger.info("[2/3] Initializing LSTM model...")
    predictor = TrajectoryPredictor(
        model_path=config.MODEL_FILE,
        norm_path=config.NORM_PARAMS_FILE
    )

    model_loaded = False
    if os.path.exists(config.MODEL_FILE):
        try:
            if predictor.load():
                model_loaded = True
                logger.info("  Loaded existing model")
        except Exception as e:
            logger.warning(f"  Could not load model: {e}")

    if not TF_AVAILABLE:
        logger.warning(
            "  TensorFlow is not installed (use Python 3.12 and requirements.txt). "
            "The LSTM ball will be a physics-based stand-in labelled 'lstm (fallback)'."
        )
    elif not model_loaded:
        logger.info("[3/3] Training new model...")
        try:
            # Get list of existing data files for augmentation
            real_data_files = [
                os.path.join(base_dir, f)
                for f in config.DATA_FILES
                if os.path.exists(os.path.join(base_dir, f))
            ]

            X, y = generate_training_data(
                num_samples=config.TRAINING_SAMPLES,
                sequence_length=config.SEQUENCE_LENGTH,
                prediction_steps=config.PREDICTION_STEPS,
                seed=config.RANDOM_SEED,
                include_real_data=True,
                real_data_files=real_data_files if real_data_files else None
            )

            predictor.build()
            predictor.train(X, y, epochs=config.TRAINING_EPOCHS, verbose=0)
            predictor.save()
            model_loaded = True
            logger.info("  Model trained and saved")

        except Exception as e:
            logger.error(f"  Model training failed: {e}")

    app_state.set_predictor(predictor, model_loaded)
    app_state.mark_initialized()

    logger.info("=" * 50)
    logger.info(f"Server ready at http://localhost:{config.PORT}")
    logger.info("=" * 50)


@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Return application initialization status."""
    return jsonify({
        'initialized': app_state.initialization_complete,
        'model_trained': app_state.model_trained,
        'sensor_loaded': len(app_state.sensor_datasets) > 0,
        'available_datasets': list(app_state.sensor_datasets.keys())
    })


@app.route('/api/datasets')
def get_datasets():
    """Return available sensor datasets."""
    datasets = {}
    for name, data in app_state.sensor_datasets.items():
        datasets[name] = {
            'name': name,
            'point_count': data.point_count,
            'duration': data.time[-1] if data.time else 0
        }
    return jsonify(datasets)


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Run trajectory simulation with physics, sensor, and LSTM models.

    Request body should match SimulationRequest schema.
    Returns SimulationResponse with trajectory data for enabled models.
    """
    try:
        # Parse and validate request
        data = request.json or {}

        try:
            req = SimulationRequest(**data)
        except Exception as e:
            return jsonify({
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': str(e)
                }
            }), 400

        # Convert to meters
        drop_height = req.dropHeight / 100.0
        ball = BALL_PRESETS.get(req.ballType, BALL_PRESETS['steel'])

        # Create a copy of the ball to modify mass if provided
        if req.mass is not None:
            import copy
            ball = copy.deepcopy(ball)
            ball.mass = req.mass

        results = {}

        # Physics simulation (always run for reference)
        physics_result = simulate_trajectory_3d(
            ball=ball,
            initial_height=drop_height,
            initial_velocity=(0.0, req.initialVelocity, 0.0),
            initial_position=(0.0, 0.0),
            dt=config.SIMULATION_DT,
            max_time=config.MAX_SIMULATION_TIME,
            include_drag=req.dragEnabled,
            restitution=req.restitution,
            gravity=req.gravity
        )
        physics_result['ballColor'] = '#3b82f6'
        physics_result['label'] = 'physics'
        physics_result['ballRadius'] = ball.radius

        if req.enablePhysics:
            results['physics'] = physics_result

        # Sensor data
        if req.enableSensor:
            sensor_result = generate_sensor_trajectory(
                req.sensorDataFile,
                drop_height,
                ball,
                physics_result
            )
            if sensor_result:
                results['sensor'] = sensor_result

        # LSTM prediction
        if req.enableLstm:
            lstm_result = generate_lstm_prediction(
                ball,
                drop_height,
                physics_result,
                req.lstmBlend
            )
            results['lstm'] = lstm_result

        return jsonify(results)

    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        return jsonify({
            'error': {
                'code': 'SIMULATION_ERROR',
                'message': str(e)
            }
        }), 500


def generate_sensor_trajectory(
    dataset_name: str,
    height: float,
    ball: Ball,
    physics_reference: dict
) -> Optional[dict]:
    """
    Generate sensor-based trajectory scaled to drop height.

    Args:
        dataset_name: Name of sensor dataset to use
        height: Drop height in meters
        ball: Ball object for properties
        physics_reference: Physics result for time alignment

    Returns:
        Trajectory dict or None if sensor data unavailable
    """
    sensor_data = app_state.get_sensor_data(dataset_name)
    if not sensor_data:
        logger.warning(f"Sensor dataset not found: {dataset_name}")
        return None

    try:
        original_max = max(sensor_data.height) if sensor_data.height else 1.0
        scale = height / original_max if original_max > 0 else 1.0

        sensor_times = np.array(sensor_data.time)
        sensor_heights = np.array([max(0, h * scale) for h in sensor_data.height])

        # Extend with decay if trajectory ends above ground
        last_height = sensor_heights[-1]
        last_time = sensor_times[-1]

        if last_height > 0.01:
            decay_time = np.linspace(last_time, last_time + 1.0, 50)
            decay_heights = last_height * np.exp(-5 * (decay_time - last_time))
            decay_heights = np.maximum(decay_heights, 0)
            sensor_times = np.concatenate([sensor_times, decay_time[1:]])
            sensor_heights = np.concatenate([sensor_heights, decay_heights[1:]])

        # Resample to physics time points
        physics_times = physics_reference['time']
        resampled_heights = np.interp(
            physics_times,
            sensor_times,
            sensor_heights,
            left=sensor_heights[0],
            right=0.0
        )

        # Align sensor data (y=0 is floor) to 3D model (y=center)
        # Add radius so that 0 becomes ball.radius
        aligned_heights = resampled_heights + ball.radius

        # Ensure non-negative (fix ground clipping)
        # The lowest valid center position is ball.radius
        aligned_heights = np.maximum(ball.radius, aligned_heights).tolist()

        # Clamp near-ground values
        for i in range(len(aligned_heights)):
            if aligned_heights[i] < ball.radius + 0.005:
                aligned_heights[i] = ball.radius

        return {
            'time': physics_times,
            'x': [-0.4] * len(physics_times),
            'y': aligned_heights,
            'z': [0.0] * len(physics_times),
            'ballRadius': ball.radius,
            'ballColor': '#10b981',
            'label': 'sensor',
            'isRigid': ball.is_rigid
        }

    except Exception as e:
        logger.error(f"Sensor trajectory error: {e}")
        return None


def generate_lstm_prediction(
    ball: Ball,
    height: float,
    physics_reference: dict,
    blend_factor: float = 0.5
) -> dict:
    """
    Generate LSTM-based trajectory prediction.

    Uses the trained LSTM model to predict trajectory, blended with
    physics for stability. Falls back to physics-based variation if
    model unavailable.

    Args:
        ball: Ball object for properties
        height: Drop height in meters
        physics_reference: Physics result for reference
        blend_factor: LSTM vs physics blend (0=physics, 1=LSTM)

    Returns:
        Trajectory dict
    """
    physics_times = physics_reference['time']
    physics_heights = physics_reference['y']
    num_points = len(physics_times)

    predictor = app_state.predictor

    if not app_state.model_trained or predictor is None:
        return create_lstm_fallback(physics_reference, height, ball, blend_factor)

    try:
        # Get normalization parameters
        time_max = predictor.norm_params.time_max
        height_max = predictor.norm_params.height_max

        # Sampling rate correction
        # Physics runs at config.SIMULATION_DT (e.g. 0.002s = 500Hz)
        # Model trained at 0.02s (50Hz)
        # We must downsample input and upsample output
        step_ratio = int(0.02 / config.SIMULATION_DT)  # Should be 10

        physics_arr = np.array(physics_heights)
        input_points = []

        # Use first N points of physics trajectory as input (downsampled)
        seq_len = predictor.sequence_length

        # We take points at indices 0, 10, 20...
        # We need seq_len points
        input_limit = min(seq_len * step_ratio, num_points)

        # Collect raw points for velocity calculation
        raw_points = []
        for i in range(0, input_limit, step_ratio):
            raw_points.append([physics_times[i], physics_arr[i]])

        # Compute velocity for each point
        for j, (t_raw, h_raw) in enumerate(raw_points):
            t_norm = t_raw / time_max
            h_norm = h_raw / height_max
            # Compute velocity using finite difference
            if j < len(raw_points) - 1:
                dt = raw_points[j + 1][0] - t_raw
                dh = raw_points[j + 1][1] - h_raw
                v = dh / dt if dt > 1e-8 else 0.0
            elif j > 0:
                dt = t_raw - raw_points[j - 1][0]
                dh = h_raw - raw_points[j - 1][1]
                v = dh / dt if dt > 1e-8 else 0.0
            else:
                v = 0.0
            # Normalize velocity
            v_norm = v / (height_max / time_max)
            input_points.append([t_norm, h_norm, v_norm])

        # Pad if needed (if simulation crashed early)
        while len(input_points) < seq_len:
            input_points.append(input_points[-1] if input_points else [0, 0, 0])

        input_seq = np.array(input_points)

        # Calculate how many steps we need to predict to cover the full duration
        # Full duration in seconds
        duration = physics_times[-1]
        # Required points at 50Hz
        required_model_steps = int(duration / 0.02) + 10 # buffer

        # Run prediction
        # predict_full returns the SEQUENCE including input + prediction
        full_pred_norm = predictor.predict_full(
            initial_sequence=input_seq,
            total_steps=max(required_model_steps, seq_len + 10),
            max_iterations=required_model_steps
        )

        # Upsample back to physics timeline (500Hz)
        # Extract components
        pred_times_norm = full_pred_norm[:, 0]
        pred_heights_norm = full_pred_norm[:, 1]

        # Denormalize using model's stored parameters for consistency
        pred_times_s = predictor.denormalize_time(pred_times_norm)
        lstm_heights_denorm = predictor.denormalize_height(pred_heights_norm)

        # Interpolate heights onto physics timestamps
        # This gives us the LSTM prediction at every physics time step
        lstm_heights_upsampled = np.interp(
            physics_times,
            pred_times_s,
            lstm_heights_denorm
        )

        # Build final blended trajectory
        full_trajectory = np.zeros(num_points)

        # Define where "prediction" effectively starts in physics time
        # This corresponds to the end of the input sequence
        # We downsampled seq_len * step_ratio points
        prediction_start_idx = len(input_points) * step_ratio
        prediction_start_idx = min(prediction_start_idx, num_points)

        # Copy the "seed" part from physics directly (ground truth)
        full_trajectory[:prediction_start_idx] = physics_arr[:prediction_start_idx]

        for i in range(prediction_start_idx, num_points):
            p_val = physics_arr[i]
            l_val = lstm_heights_upsampled[i]

            # Ensure non-negative
            l_val = max(0, l_val)

            # Blend
            blended_val = p_val * (1 - blend_factor) + l_val * blend_factor
            full_trajectory[i] = max(0, min(blended_val, height * 1.2))

        return {
            'time': physics_times,
            'x': [0.4] * num_points,
            'y': full_trajectory.tolist(),
            'z': [0.0] * num_points,
            'ballRadius': ball.radius,
            'ballColor': '#f43f5e',
            'label': 'lstm',
            'isRigid': ball.is_rigid
        }

    except Exception as e:
        logger.error(f"LSTM prediction failed: {e}", exc_info=True)
        return create_lstm_fallback(physics_reference, height, ball, blend_factor)


def create_lstm_fallback(
    physics_reference: dict,
    height: float,
    ball: Ball,
    blend_factor: float = 0.5
) -> dict:
    """
    Create fallback LSTM-like trajectory when model unavailable.

    Uses physics as base but adds slight "AI-like" imperfections/smoothing
    to distinguish it from pure physics, without breaking the laws of gravity
    (unlike the previous random noise version).
    """
    times = physics_reference['time']
    heights = np.array(physics_reference['y'])
    num_points = len(times)

    # Simple smoothing of physics to look like an approximated prediction
    # This creates a "perfect" but slightly floaty look
    varied_heights = []

    # Physics is our best guess if model fails
    # Use low-frequency sine waves (Perlin-like) instead of white noise for realism
    np.random.seed(config.RANDOM_SEED)

    # Random parameters for "AI drift"
    phase_shift = np.random.uniform(0, 2*np.pi)
    freq_error = np.random.uniform(0.9, 1.1)

    for i, h in enumerate(heights):
        t = times[i]

        # Smooth drift + slight high-freq wobble
        # Lower frequency for smoother "AI error" look
        drift = 0.02 * np.sin(2 * np.pi * 0.3 * t + phase_shift) * blend_factor
        wobble = 0.003 * np.sin(2 * np.pi * 3.0 * t) * blend_factor

        # Combined noise
        noise = drift + wobble

        # Don't go through floor (clamp to radius)
        new_h = max(ball.radius, h + noise)
        varied_heights.append(new_h)

    return {
        'time': times,
        'x': [0.4] * num_points,
        'y': varied_heights,
        'z': [0.0] * num_points,
        'ballRadius': ball.radius,
        'ballColor': '#f43f5e',
        'label': 'lstm (fallback)',
        'isRigid': ball.is_rigid
    }


@app.route('/api/ball-types')
def get_ball_types():
    """Return available ball type presets."""
    balls = {}
    for key, ball in BALL_PRESETS.items():
        balls[key] = {
            'name': ball.name,
            'mass': ball.mass * 1000,  # Convert to grams
            'radius': ball.radius * 100,  # Convert to cm
            'material': ball.material,
            'color': ball.color,
            'restitution': ball.restitution,
            'isRigid': ball.is_rigid,
            'associatedDataset': ball.associated_dataset
        }
    return jsonify(balls)


# Initialize on import
with app.app_context():
    initialize_app()


if __name__ == '__main__':
    app.run(
        debug=False,
        host='0.0.0.0',
        port=config.PORT,
        threaded=True
    )
