# Comprehensive Analysis of Trajectory Prediction System

## 1️⃣ Global Project Understanding

**Project Goal:**
The primary objective of this project is to simulate, predict, and visualize the trajectory of a bouncing ball using three distinct approaches simultaneously:
1.  **Theoretical Physics:** A deterministic model based on Newtonian mechanics.
2.  **Experimental Data:** Real-world measurements collected via an ultrasonic sensor.
3.  **Machine Learning:** A Long Short-Term Memory (LSTM) neural network trained to predict future positions.

**Problem Solved:**
The system addresses the challenge of modeling complex dynamic systems where theoretical idealizations (perfect elasticity, no air resistance) often diverge from reality. By juxtaposing a theoretical model, real sensor data, and an AI prediction, the application demonstrates the strengths and limitations of each approach in capturing the physics of a damping harmonic oscillator (a bouncing ball losing energy).

**Software Architecture:**
The project follows a **Client-Server architecture**:
*   **Backend (Flask/Python):** Acts as the computational brain. It handles physics calculations (`physics.py`), processes raw sensor data (`data_processing.py`), and serves the AI model (`model.py`).
*   **Frontend (HTML/JS/Three.js):** Acts as the visualization layer. It renders a 3D environment and 2D charts to allow users to intuitively compare the three trajectories.

**Data Pipeline:**
1.  **Acquisition:** An ultrasonic sensor measures the distance to a ball dropping from a fixed height.
2.  **Preprocessing:** Raw CSV data (Time vs. Distance) is cleaned (outliers removed), inverted (Distance $\to$ Height), smoothed (Savitzky-Golay filter), and resampled to a uniform time grid.
3.  **Modeling/Simulation:**
    *   *Physics:* Numerical integration of equations of motion.
    *   *AI:* The preprocessed data feeds into an LSTM network to predict future time steps.
4.  **Visualization:** The API sends JSON trajectories to the frontend, where Three.js renders the 3D scene and Chart.js plots the height-vs-time graph.

---

## 2️⃣ Codebase Structure

### Core Application
*   **`app.py`**: The central controller. It initializes the Flask server, loads the LSTM model and sensor data into memory, and exposes the `/api/simulate` endpoint. It orchestrates the flow: receiving user parameters $\to$ running physics engine $\to$ processing sensor data $\to$ generating AI predictions $\to$ returning a unified JSON response.
*   **`config.py`**: Centralized configuration (constants like Gravity $g=9.81$, Air Density $\rho=1.25$, sampling rates). Ensures consistency across modules.
*   **`schemas.py`**: Defines strict data contracts (interfaces) using **Pydantic**. It ensures that the API receives valid inputs (e.g., mass > 0, restitution between 0 and 1).

### Physics & Math Engines
*   **`physics.py`**: The physics engine. It defines the `Ball` class and the `simulate_trajectory_3d` function, which performs the time-stepping simulation.
*   **`data_processing.py`**: Handles the "messy" real world. Contains logic to load CSVs, interpolate missing values (handling sensor "glitches"), and convert sensor distance to ball height ($h = H_{sensor} - d$).

### Machine Learning
*   **`model.py`**: Defines the Neural Network architecture. It builds a Bidirectional LSTM using TensorFlow/Keras, handles normalization (scaling inputs to 0-1 range), and manages the prediction loop.
*   **`synthetic.py`**: A data generator. Since real-world data is limited, this script uses the physics engine to generate thousands of "fake" but realistic trajectories (varying height, mass, bounciness) to train the LSTM model.

### Visualization
*   **`templates/index.html`**: A monolithic frontend file containing the HTML structure, CSS styling, and complex JavaScript logic for the Three.js 3D scene and Chart.js graphs.

---

## 3️⃣ Data & Datasets

### Experimental Collection
The data files (`data.csv`, `data2.csv`, `data3.csv`) represent time-series measurements from an ultrasonic sensor mounted on a stand ("potence"), looking down at a bouncing ball.

**Structure:**
*   **`time_after_drop_ms`**: Time in milliseconds since the recording started.
*   **`distance_cm`**: The measured distance from the *sensor* to the *top of the ball*.

**Mathematical Transformation:**
To get the physical height of the ball $h(t)$, the code applies:
$$ h(t) = H_{sensor} - d(t) $$
Where $H_{sensor}$ is the fixed height of the sensor (configured as 70.0 cm in `config.py`).

### Dataset Comparison
*   **`data.csv`**: Appears to be the "Golden Master". The trajectory is clean, showing a classic damped bouncing pattern.
*   **`data2.csv` & `data3.csv`**: Represent subsequent experiments. These files likely contain more noise or different initial conditions. The `data_processing.py` module includes specific logic (`clean_sensor_data`) to handle "glitches" where the sensor returns a max reading (e.g., 66.0 cm) indicating it lost track of the ball. These outliers are replaced using linear interpolation.

---

## 4️⃣ Physical Model

The simulation in `physics.py` relies on **Classical Newtonian Mechanics**.

### 1. Physical Assumptions
*   **Point Mass:** The ball is modeled as a point mass $m$, ignoring rotation (magnus effect) and internal deformation dynamics (though visual squash/stretch is applied).
*   **Vertical Motion:** The primary motion is along the Y-axis. The X/Z motion in 3D is for visualization separation only.
*   **Forces:** Only Gravity and Aerodynamic Drag are considered.

### 2. Forces & Equations of Motion
The ball is subject to two forces:
1.  **Gravity ($F_g$):** Constant downward force.
    $$ \vec{F}_g = m \vec{g} = (0, -mg, 0) $$
2.  **Aerodynamic Drag ($F_d$):** Resistance from the air, opposing the velocity vector $\vec{v}$.
    $$ \vec{F}_d = -\frac{1}{2} \rho C_d A |\vec{v}| \vec{v} $$
    *   $\rho$: Air density ($1.25 \, \text{kg/m}^3$).
    *   $C_d$: Drag coefficient (0.47 for a sphere).
    *   $A$: Cross-sectional area ($\pi r^2$).

Applying Newton's Second Law ($\sum \vec{F} = m \vec{a}$):
$$ \vec{a} = \vec{g} - \frac{\rho C_d A}{2m} |\vec{v}| \vec{v} $$

### 3. Numerical Integration (Velocity Verlet)
To solve these differential equations, the code uses the **Velocity Verlet** integrator. This is chosen over Euler integration because it is Symplectic (preserves energy better over time).

$$ \vec{r}(t+\Delta t) = \vec{r}(t) + \vec{v}(t)\Delta t + \frac{1}{2}\vec{a}(t)\Delta t^2 $$
$$ \vec{v}(t+\Delta t) = \vec{v}(t) + \frac{\vec{a}(t) + \vec{a}(t+\Delta t)}{2}\Delta t $$

### 4. Impact Model & Coefficient of Restitution
When the ball hits the ground ($y \le r$), an inelastic collision occurs. The energy loss is governed by the **Coefficient of Restitution ($e$)**.

#### Derivation of $e$
The coefficient $e$ is defined as the ratio of relative speeds after and before impact:
$$ e = \frac{|v_{after}|}{|v_{before}|} $$

Let's derive the relationship with height.
From Conservation of Energy (neglecting air drag for the moment of impact):
$$ mgh = \frac{1}{2}mv^2 \implies v = \sqrt{2gh} $$

1.  **Just before impact $n$:** The ball falls from height $h_n$.
    $$ v_{down} = \sqrt{2gh_n} $$
2.  **Just after impact $n$:** The ball rebounds with velocity $v_{up}$.
    $$ v_{up} = e \cdot v_{down} = e\sqrt{2gh_n} $$
3.  **Peak of rebound $n$:** The ball rises to height $h_{n+1}$.
    $$ v_{up} = \sqrt{2gh_{n+1}} $$

Equating the two expressions for $v_{up}$:
$$ \sqrt{2gh_{n+1}} = e\sqrt{2gh_n} $$
Squaring both sides:
$$ 2gh_{n+1} = e^2 (2gh_n) $$
$$ h_{n+1} = e^2 h_n $$

Solving for $e$:
$$ e = \sqrt{\frac{h_{n+1}}{h_n}} $$

**Why the square root?**
The square root appears because the coefficient of restitution $e$ is a ratio of **velocities** (momentum), whereas height is a measure of **potential energy**. Since kinetic energy scales with the square of velocity ($K \propto v^2$) and potential energy scales linearly with height ($U \propto h$), the ratio of heights corresponds to the square of the ratio of velocities ($e^2$).

#### Geometric Sequence of Heights
From the relation $h_{n+1} = e^2 h_n$, we can see that the maximum height of each bounce follows a geometric progression.
Let $r = e^2$ be the common ratio.
*   Bounce 0 (Drop): $h_0$
*   Bounce 1: $h_1 = e^2 h_0$
*   Bounce 2: $h_2 = e^2 h_1 = e^2 (e^2 h_0) = e^4 h_0$
*   Bounce $n$: $h_n = e^{2n} h_0$

This explains the characteristic "staircase" decay of the log of the height, or the exponential decay envelope seen in the Time-Height graph.

**Note on Reality:** The code actually implements a *decaying* restitution to model structural fatigue: $e_{eff} = e \cdot (0.95)^{n}$. This makes the real decay slightly faster than a pure geometric sequence.

---

## 5️⃣ LSTM / Machine Learning Model

The project uses a **Recurrent Neural Network (RNN)**, specifically an **LSTM (Long Short-Term Memory)**, which is ideal for time-series data because it maintains an internal "memory" of previous states to predict the future.

### Why LSTM?
Standard feed-forward networks assume inputs are independent. A bouncing ball trajectory is highly dependent on its history (velocity, position, phase). The LSTM cell's gating mechanisms (forget gate, input gate, output gate) allow it to learn long-term dependencies, such as the periodic nature of the bounces and the rate of energy decay.

### Architecture (`model.py`)
1.  **Input Layer:** Shape `(Batch, 50, 3)`. Sequence length = 50 steps.
    *   Features: `[Normalized Time, Normalized Height, Normalized Velocity]`
2.  **Bidirectional LSTM (128 units):** Processes the sequence forwards and backwards. This is powerful for trajectory interpolation but in a real-time prediction scenario, it primarily helps the model understand the local "shape" of the curve better.
3.  **Dropout (0.2):** Regularization to prevent overfitting to the synthetic training data.
4.  **LSTM (64 units):** A second unidirectional layer to condense the temporal features.
5.  **Dense Layers:**
    *   Dense(64, activation='relu')
    *   Dense(20, activation='linear') -> Outputting 10 steps of (time, height).

### Training Process
*   **Synthetic Data Generation:** `synthetic.py` acts as a simulator. It uses the physics engine to generate thousands of trajectories with randomized $g \in [9.7, 9.9]$, $m \in [0.01, 0.1]$, and $e \in [0.6, 0.9]$. This forces the LSTM to learn the *general laws* of bouncing physics rather than memorizing a specific dataset.
*   **Normalization:** Inputs are strictly scaled to [0,1] based on `TIME_MAX=3.0s` and `HEIGHT_MAX=2.0m`.
*   **Loss Function:** Mean Squared Error (MSE).

---

## 6️⃣ Simulations

### 3D Simulation (Three.js)
The 3D view renders a "digital twin" of the experiment.
*   **Physics Ball (Blue):** Follows the mathematical equations exactly.
*   **Sensor Ball (Green):** Replays the recorded CSV data.
*   **AI Ball (Red):** Shows the LSTM's prediction.
The balls are squashed vertically upon impact (`scale.set(1/s, s, 1/s)`) to visualize elasticity, a nice detail adding to the realism.

### 2D Graph (Chart.js)
Plots Height (y-axis) vs. Time (x-axis). This visualization clearly shows the **Envelope of Decay**. In a perfect vacuum with constant $e$, the peaks would follow an exponential decay curve $y = A e^{-\lambda t}$. The graph allows the user to see how air drag (in the physics model) and real-world energy loss (in sensor data) cause deviation from the ideal curve.

---

## 7️⃣ Application Layer

**Tech Stack:**
*   **Backend:** Python 3.9+, Flask (Web Framework), NumPy (Math), Pandas (Data), TensorFlow (AI).
*   **Frontend:** HTML5, CSS3 (Modern Dark Mode), JavaScript (ES6+).

**Interaction Flow:**
1.  User adjusts sliders (Mass, Gravity, Restitution) on the web interface.
2.  Clicking "Simulate" sends a POST request to `/api/simulate`.
3.  The backend runs the `simulate_trajectory_3d` function with the new parameters.
4.  If "Sensor" is enabled, it resamples the CSV data to match the physics time-steps.
5.  If "LSTM" is enabled, it runs the neural network inference.
6.  The frontend receives the arrays and updates the Three.js scene and Chart.js datasets.

---

## 8️⃣ Libraries & Dependencies

*   **Flask:** Lightweight web server. Chosen for simplicity in serving the API and static files.
*   **NumPy:** The backbone of scientific computing in Python. Used for efficient vector operations (arrays, velocities, forces).
*   **Pandas:** Used in `data_processing.py` for reading CSVs and handling time-series data (interpolation, filtering).
*   **TensorFlow/Keras:** The Deep Learning framework used to build and train the LSTM model.
*   **Three.js:** The industry standard for WebGL 3D graphics in the browser.
*   **Chart.js:** A popular library for rendering responsive, interactive 2D charts.

---

## 9️⃣ Critical Analysis

### Strengths
1.  **Hybrid Approach:** Combining Physics, Sensor Data, and AI provides a complete pedagogical view of the problem.
2.  **Robust Physics:** The implementation of air drag and velocity-dependent drag is physically accurate.
3.  **Data Handling:** The interpolation logic (`data_processing.py`) effectively sanitizes noisy sensor data.

### Weaknesses & Limitations
1.  **Sensor Resolution:** The ultrasonic sensor has a limited sampling rate (~50Hz estimated from CSV), which can miss the exact moment of impact.
2.  **Physics Simplification:** The model assumes a constant drag coefficient ($C_d$). In reality, $C_d$ changes with Reynolds number (velocity).
3.  **LSTM Generalization:** If the user sets "alien" parameters (e.g., gravity = 20 m/s²), the LSTM (trained on Earth-like physics) might produce erratic predictions.

### Improvements
*   **Sensor Fusion:** Combine accelerometer data (if available) with the ultrasonic sensor using a Kalman Filter for smoother position tracking.
*   **Variable Drag:** Implement a dynamic $C_d$ based on velocity to model the transition between laminar and turbulent flow.
*   **Online Learning:** Allow the LSTM to "learn" from the user's specific sensor data in real-time, adapting its weights to the specific ball being used.
