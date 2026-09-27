# 🎯 Trajectory Prediction System

A professional web application for simulating and predicting ball trajectories using **three simultaneous models** displayed side-by-side in real-time 3D.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Three.js](https://img.shields.io/badge/Three.js-r128-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-red.svg)

## ✨ Features

- 🎮 **3 Balls, 3 Models** - Compare Physics, Sensor, and LSTM predictions simultaneously
- 🌐 **Real-time 3D** - Interactive Three.js visualization with orbit controls
- 📊 **2D Charts** - Chart.js trajectory comparison graphs
- 🧠 **LSTM Neural Network** - Machine learning trajectory prediction
- 📐 **Physics Engine** - Accurate simulation with air drag and bouncing
- 📡 **Real Sensor Data** - Process experimental measurements from CSV
- 🌍 **Bilingual** - English 🇬🇧 / French 🇫🇷 toggle
- 🎨 **Dark/Light Themes** - Modern, professional UI

## 🎮 The Three Balls

| Ball | Color | Position | Model |
|------|-------|----------|-------|
| 🔵 Blue | `#3b82f6` | Center | **Physics** - Newtonian mechanics with drag |
| 🟢 Green | `#10b981` | Left | **Sensor** - Real experimental data from CSV |
| 🔴 Red | `#f43f5e` | Right | **LSTM** - Neural network prediction |

All three balls drop simultaneously, allowing direct visual comparison of each prediction method.

## 📁 Project Structure

```
trajectory_project/
├── app.py              # Flask web server
├── physics.py          # 3D physics simulation
├── data_processing.py  # Sensor data handling
├── synthetic.py        # Training data generator
├── model.py            # LSTM neural network
├── templates/
│   └── index.html      # Web UI (Three.js + Chart.js)
├── data.csv            # Real sensor data
├── model.keras         # Trained model (auto-generated)
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors numpy pandas scipy tensorflow
```

### 2. Run the Application

```bash
cd trajectory_project
python app.py
```

### 3. Open in Browser

Navigate to **http://localhost:5000**

## 🎮 Using the Application

### Controls

| Control | Description |
|---------|-------------|
| **Ball Type** | Select from Steel, Rubber, Ping-Pong, Tennis, or Custom |
| **Drop Height** | Adjust from 10cm to 200cm |
| **Model Toggles** | Enable/disable Physics, Sensor, LSTM predictions |
| **SIMULATE** | Run the simulation |

### 3D View Controls

- **Left-click + Drag** - Rotate camera
- **Scroll** - Zoom in/out

### Views

1. **3D Simulation** - Watch the ball drop in real-time 3D
2. **2D Graph** - Compare all trajectories on a time-height chart
3. **Statistics** - View fall time, max velocity, bounce count

## 🧠 The Three Models

### 1. Physics Model (Blue 🔵)
- Newtonian mechanics with air drag
- Coefficient of restitution for bouncing
- Real-time numerical integration

### 2. Sensor Data (Green 🟢)
- Real experimental measurements from `data.csv`
- Automatic noise filtering and interpolation
- Scales to match selected drop height

### 3. LSTM Prediction (Red 🔴)
- Trained on 800 synthetic trajectories
- Sequence-to-sequence prediction
- Captures bouncing dynamics

## 🎨 Screenshots

### Dark Theme
```
┌─────────────────────────────────────────────────────────┐
│  🎯 Trajectory Simulation           🇬🇧 🇫🇷  ☀️ 🌙   │
├─────────────┬───────────────────────────────────────────┤
│ Ball Type   │                                           │
│ ┌─────────┐ │         ╭──────────────────╮             │
│ │ Steel ▼ │ │        /                    \            │
│ └─────────┘ │       /    3D SIMULATION     \           │
│             │      │                        │           │
│ Parameters  │      │     🔴 ← Ball          │           │
│ Height: 67cm│      │      │                 │           │
│ ═══════○═══ │       \     ↓ Path           /           │
│             │        ╰──────────────────────╯           │
│ ☑ Physics   │      ════════════════════════            │
│ ☑ Sensor    │              Ground                       │
│ ☑ LSTM      │                                           │
│             │  [ 3D Simulation ] [ 2D Graph ] [ Stats ] │
│ ┌─────────┐ │                                           │
│ │▶ SIMULATE│ │                                           │
│ └─────────┘ │                                           │
└─────────────┴───────────────────────────────────────────┘
```

## 📊 API Endpoints

### `POST /api/simulate`

Run a simulation with given parameters.

**Request:**
```json
{
  "dropHeight": 67,
  "ballType": "steel",
  "enablePhysics": true,
  "enableSensor": true,
  "enableLstm": true
}
```

**Response:**
```json
{
  "physics": {
    "time": [0, 0.005, ...],
    "x": [0, 0, ...],
    "y": [0.67, 0.669, ...],
    "z": [0, 0, ...]
  },
  "sensor": {
    "time": [...],
    "height": [...]
  },
  "lstm": {
    "time": [...],
    "height": [...]
  }
}
```

### `GET /api/ball-types`

Get available ball presets.

## ⚙️ Configuration

Edit constants in `app.py`:

```python
RANDOM_SEED = 42      # For reproducibility
DATA_FILE = "data.csv"  # Sensor data file
MODEL_FILE = "model.keras"  # Saved model
```

## 🔧 Troubleshooting

### Model Training Takes Too Long
The first run trains the LSTM model (~1-2 minutes). Subsequent runs load the saved model.

### Port Already in Use
Change the port in `app.py`:
```python
app.run(port=8080)  # Use different port
```

### TensorFlow Warnings
These are suppressed by default. Safe to ignore GPU-related messages.

## 📜 License

MIT License - Free to use and modify!

---

**Made with ❤️ using Flask, Three.js, Chart.js, and TensorFlow**
