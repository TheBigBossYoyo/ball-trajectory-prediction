# Ball Trajectory Prediction

I built a small web app that drops a virtual ball three times at once, using three different ways of knowing where the ball will go: a physics simulation, a real sensor recording, and a neural network. All three are rendered side by side in 3D so you can see where they agree and where they don't.

This started as a physics project (the `PEN` folder it lives in) about bouncing balls and energy loss on impact. I had real data from an ultrasonic sensor recording a ball falling and bouncing, and I wanted to compare that to what a textbook physics model predicts, and to what a machine learning model would predict if it only saw the shape of the curve.

## What it does

You pick a ball (steel, rubber, ping-pong, tennis, or custom mass/size) and a drop height, and the app simulates the fall with three models simultaneously:

- Physics (blue ball): a numerical integration of Newton's second law with air drag and a bouncing model.
- Sensor (green ball): real measurements I recorded with an ultrasonic distance sensor, replayed and rescaled to the chosen height.
- LSTM (red ball): a recurrent neural network trained on a mix of simulated and real trajectories, predicting the next positions from the recent past.

The three balls drop together in a Three.js 3D scene, there's a 2D height-vs-time chart (Chart.js) to compare curves precisely, and a stats panel with fall time, max velocity, and bounce count. The interface has an English/French toggle and a light/dark theme, both of which I added mostly because I was switching between languages while writing the report.

## How it works

### Physics model

The physics engine (`physics.py`) treats the ball as a point mass under gravity and quadratic air drag, integrated with **velocity Verlet** rather than simple Euler steps, because Verlet conserves energy much better over many bounces:

$$\vec{a} = \vec{g} - \frac{\rho C_d A}{2m}\,|\vec{v}|\,\vec{v}$$

$$\vec{r}(t+\Delta t) = \vec{r}(t) + \vec{v}(t)\Delta t + \tfrac{1}{2}\vec{a}(t)\Delta t^2, \qquad \vec{v}(t+\Delta t) = \vec{v}(t) + \tfrac{\vec{a}(t)+\vec{a}(t+\Delta t)}{2}\Delta t$$

with $\rho = 1.25\ \text{kg/m}^3$ for air density and $C_d = 0.47$ for a sphere. Each ball preset (steel, rubber, ping-pong, tennis) has its own mass, radius and coefficient of restitution $e$, taken from real measurements where I had them (the rubber ball preset uses 59 g and matches `data.csv`).

When the ball hits the ground, its velocity flips and shrinks by $e$. From energy conservation ($mgh = \tfrac12 mv^2$), the rebound height after bounce $n$ follows a geometric sequence:

$$e = \sqrt{\dfrac{h_{n+1}}{h_n}} \qquad\Rightarrow\qquad h_n = e^{2n} h_0$$

which is why the peaks of a bouncing-ball chart decay along an exponential envelope, not linearly. The square comes from the fact that $e$ is a ratio of velocities while $h$ measures energy, and kinetic energy scales as $v^2$. In the code I also decay the restitution itself a little on every bounce, $e_\text{eff} = e \cdot 0.95^n$, to model the ball losing more energy as it deforms repeatedly. A pure constant-$e$ model bounces for too long compared to what I actually recorded.

### Sensor data

The sensor readings (`data.csv`, `data2.csv`, `data3.csv`) are two-column CSVs: time in milliseconds and the distance from the sensor (mounted 70 cm above the floor) to the top of the ball, sampled roughly every 8-10 ms. `data_processing.py` cleans this up: readings above 66 cm mean the sensor lost the ball (it clamps a long trailing run of these to "ball at rest," and linearly interpolates isolated glitches), then converts distance to height ($h = H_\text{sensor} - d$), applies a Savitzky-Golay smoothing filter, and resamples to a uniform 10 ms grid so it can be compared against the other two models on the same time axis.

### LSTM model

The network (`model.py`) is a small recurrent model built with Keras: a bidirectional LSTM (128 units) followed by dropout, a second unidirectional LSTM (64 units), dropout again, a dense layer (64, ReLU), and a final dense layer outputting 10 future (time, height) pairs. It takes the last 50 timesteps of normalized time, height, and velocity as input and predicts the next 10 steps, which get fed back in autoregressively to build a full trajectory.

Real recordings alone aren't enough data to train a network, so `synthetic.py` generates several hundred additional trajectories by running the physics engine itself with randomized mass, gravity, and restitution, mixed in with augmented copies of the real recordings (800 training sequences total by default). This is meant to teach the network the general shape of bouncing-ball motion rather than memorizing one specific fall. Training uses Adam with gradient clipping, mean squared error loss, and early stopping on a validation split.

Because the network only ever sees Earth-like gravity and everyday ball masses during training, it can behave oddly if you push the physics sliders to extreme values it never saw in training: it's approximating a pattern, not solving the equations.

## What I found

Fitting the restitution coefficients to the recorded data (rather than using idealized textbook values) mattered a lot: the steel ball needed a restitution around 0.6 and the added per-bounce decay ($0.95^n$) to match how quickly it actually stopped bouncing, whereas a constant, higher restitution made it bounce for much longer than the real ball did. The rubber ball preset (59 g, restitution 0.85) was tuned specifically against `data.csv` and tracks it closely. The physics and sensor curves generally agree on the overall decay envelope; where they diverge is mostly right at each impact, since the sensor's sampling rate can miss the exact instant of the bounce and smooths the peak slightly.

I wrote up a longer analysis of the codebase and the physics in more detail in [`ANALYSIS.md`](ANALYSIS.md) (English) and [`RAPPORT_ANALYSE.md`](RAPPORT_ANALYSE.md) (French version of the same report), and a shorter side-by-side comparison of the physics and LSTM approaches in [`EXPLANATION.txt`](EXPLANATION.txt).

## Running it

```bash
pip install -r requirements.txt
```

`requirements.txt` covers Flask, NumPy, Pandas, SciPy and Pydantic. TensorFlow is commented out there, since it isn't yet compatible with newer Python versions, so to get the LSTM ball working you'll also need:

```bash
pip install tensorflow
```

Without TensorFlow installed, the app still runs and shows the physics and sensor balls; the LSTM toggle is just disabled. Then:

```bash
python app.py
```

and open `http://localhost:5000`. On first run it trains the LSTM (about 1-2 minutes); after that it loads the saved `model.keras`.

## Limitations and what I'd do next

The drag coefficient is treated as constant, when in reality it depends on the Reynolds number and would change a bit over the course of a fall. The ultrasonic sensor's sampling rate limits how precisely the exact bounce moment is captured, which softens the sharp corners you'd expect at each impact. The LSTM is also only as good as its training distribution: it learns "how bouncing balls generally behave," not the underlying differential equation, so it can't be trusted far outside the range of masses, heights, and gravities it was trained on. If I kept working on this, I'd want to try a Kalman filter to fuse sensor data with an accelerometer for better tracking, a velocity-dependent drag coefficient, and possibly letting the model fine-tune on new sensor data instead of only using the fixed synthetic set.
