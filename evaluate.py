"""
Quantitative comparison of the physics model and the LSTM against my own
sensor recordings.

Run:
    python evaluate.py                # physics + LSTM (needs TensorFlow, ~10 min on CPU)
    python evaluate.py --skip-lstm    # physics only (seconds)

What is compared
----------------
Measured curve: the height from data_processing.process_sensor_csv, in metres,
as the height of the top of the ball above the floor (the sensor looks down at
the top of the ball). The models are converted to the same quantity (centre
height + ball radius). Errors are scored only at samples that sit next to a
real sensor reading; samples the cleaning step filled by interpolation are not
scored, since they are guesses and not measurements.

Physics model: the rubber preset (59 g, 3 cm radius), with only two numbers
fitted, the restitution e and the per-bounce decay factor. They are fitted on
one dataset by grid search (minimum RMSE) and then applied unchanged to the
others. Fitting on each dataset in turn gives the full cross-fit table. The
sensor cannot see closer than about 3 cm, so the ball is already moving at the
first reading; the initial downward speed is therefore estimated for every
dataset from its first 0.2 s (the same prefix the LSTM is given) and is not
fitted on the scored part.

LSTM: for each measured dataset a fresh network is trained on the synthetic
trajectories plus augmented copies of the *other two* recordings only, so the
dataset being scored was never seen in training. It is then given the first
0.2 s of the measured fall and rolls forward on its own predictions. The
physics model gets only the drop height and is scored on the same window
(t >= 0.2 s), so both are compared over the same samples.

Outputs: docs/results.csv, docs/results.md, docs/physics_cross_fit.csv and
PNG plots in docs/figures/.
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config
from data_processing import (
    load_sensor_data, process_sensor_csv, valid_reading_mask,
)
from metrics import rmse, mae, first_crossing_time, window_apex
from physics import BALL_PRESETS, simulate_trajectory_3d

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
FIGS = os.path.join(DOCS, "figures")

DATASETS = ["data.csv", "data2.csv", "data3.csv"]
FIT_DATASET = "data.csv"       # headline physics fit
SEED_SECONDS = 0.2             # measured prefix given to the LSTM; scoring starts here
APEX_WINDOW = (0.35, 1.0)      # where to look for the first rebound peak (s)
DESCENT_LEVEL = 0.20           # height used for the "time to fall to 20 cm" check (m)
BALL = BALL_PRESETS["rubber"]

E_GRID = np.arange(0.50, 0.9501, 0.01)
DECAY_GRID = np.arange(0.80, 1.0001, 0.02)

COLORS = {"physics": "#0072B2", "lstm": "#D55E00", "constant": "0.6"}
LABELS = {
    "physics": f"physics (fitted on {FIT_DATASET})",
    "lstm": "LSTM (not trained on this file)",
    "constant": "constant = mean of measured",
}


def load_measured(name):
    """Processed measured curve plus a mask of samples backed by a real reading."""
    path = os.path.join(HERE, name)
    t, _, h = process_sensor_csv(path)
    raw = load_sensor_data(path)
    mask = valid_reading_mask(raw["time_ms"].values, raw["distance_cm"].values, t)
    ds = {"name": name, "t": t, "h": h, "mask": mask}
    ds["v0"] = estimate_v0(ds)
    return ds


def physics_curve(t_grid, restitution, decay, h0, v0=0.0, ball=BALL):
    """Top-of-ball height from the physics engine, sampled on t_grid (v0 = initial downward speed)."""
    sim = simulate_trajectory_3d(
        ball,
        initial_height=h0 - ball.radius,
        initial_velocity=(0.0, -v0, 0.0),
        restitution=restitution,
        restitution_decay=decay,
        max_time=float(t_grid[-1]) + 0.05,
    )
    return np.interp(t_grid, sim["time"], np.array(sim["y"]) + ball.radius)


def estimate_v0(ds):
    """
    Initial downward speed, estimated from the first SEED_SECONDS of the
    recording only (the same prefix the LSTM is given). The sensor cannot see
    closer than about 3 cm, so the ball is already moving when the first
    reading arrives and "released from rest at t = 0" would be wrong.
    """
    t, h, m = ds["t"], ds["h"], ds["mask"]
    sel = m & (t < SEED_SECONDS)
    res = minimize_scalar(
        lambda v0: rmse(physics_curve(t[sel], 0.8, 1.0, h[0], v0)[:len(t[sel])], h[sel]),
        bounds=(0.0, 3.0), method="bounded")
    return float(res.x)


def fit_physics(ds):
    """
    Grid search for (restitution, decay) with the lowest RMSE on one dataset,
    scored over t >= SEED_SECONDS, using that dataset's own estimated v0.
    """
    t, h, m = ds["t"], ds["h"], ds["mask"]
    sel = m & (t >= SEED_SECONDS)
    v0 = ds["v0"]
    best = (None, None, np.inf)
    for e in E_GRID:
        for k in DECAY_GRID:
            err = rmse(physics_curve(t, e, k, h[0], v0)[sel], h[sel])
            if err < best[2]:
                best = (float(e), float(k), err)
    return {"restitution": round(best[0], 3), "decay": round(best[1], 3),
            "fit_rmse_m": best[2], "v0": round(v0, 3)}


def lstm_curves(datasets, epochs):
    """
    Leave-one-dataset-out LSTM predictions. Returns ({name: curve on that
    dataset's time grid}, note). Returns (None, message) when TensorFlow is
    missing.
    """
    from model import TF_AVAILABLE, TrajectoryPredictor
    if not TF_AVAILABLE:
        return None, ("TensorFlow is not installed (it needs Python 3.9 to 3.12). "
                      "Skipping the LSTM; see the README for setup.")
    import tensorflow as tf
    from synthetic import generate_training_data, set_seed

    out = {}
    for ds in datasets:
        others = [os.path.join(HERE, d) for d in DATASETS if d != ds["name"]]
        print(f"  LSTM: training without {ds['name']} (using {len(others)} real files)...",
              flush=True)
        set_seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)
        X, y = generate_training_data(
            num_samples=config.TRAINING_SAMPLES, seed=config.RANDOM_SEED,
            include_real_data=True, real_data_files=others)
        pred = TrajectoryPredictor()
        pred.build()
        hist = pred.train(X, y, epochs=epochs, verbose=0)
        print(f"    trained {len(hist.history['loss'])} epochs, "
              f"val_loss {hist.history['val_loss'][-1]:.5f}", flush=True)

        # Seed: first SEED_SECONDS of the measured fall at 50 Hz, start-padded
        step = 0.02
        n_seed = int(round(SEED_SECONDS / step))
        grid20 = np.arange(n_seed) * step
        h20 = np.interp(grid20, ds["t"], ds["h"])
        tn = grid20 / pred.norm_params.time_max
        hn = h20 / pred.norm_params.height_max
        feats = pred._with_velocity(np.column_stack([tn, hn]))
        pad = np.tile(feats[0:1], (pred.sequence_length - n_seed, 1))
        seq = np.vstack([pad, feats])

        n_future = int(np.ceil(ds["t"][-1] / step)) + 5
        full = pred.predict_full(seq, total_steps=pred.sequence_length + n_future,
                                 max_iterations=n_future)
        real_rows = full[pred.sequence_length - n_seed:]
        t_s = real_rows[:, 0] * pred.norm_params.time_max
        h_s = real_rows[:, 1] * pred.norm_params.height_max
        out[ds["name"]] = np.interp(ds["t"], t_s, h_s)
    return out, (f"trained once per held-out dataset on synthetic data plus augmented copies of "
                 f"the other two recordings, {epochs} epochs max, seed {config.RANDOM_SEED}.")


def score(ds, curve):
    """All metrics for one model curve on one dataset, over t >= SEED_SECONDS."""
    t, h, m = ds["t"], ds["h"], ds["mask"]
    sel = m & (t >= SEED_SECONDS)
    row = {
        "n_scored": int(sel.sum()),
        "rmse_m": rmse(curve[sel], h[sel]),
        "mae_m": mae(curve[sel], h[sel]),
    }
    # Timing, with measured values taken from real-reading samples only
    tm, hm = t[m], h[m]
    meas_t20 = first_crossing_time(tm, hm, DESCENT_LEVEL)
    mod_t20 = first_crossing_time(t, curve, DESCENT_LEVEL)
    row["fall_to_20cm_err_s"] = (None if meas_t20 is None or mod_t20 is None
                                 else mod_t20 - meas_t20)
    meas_ap = window_apex(tm, hm, *APEX_WINDOW)
    mod_ap = window_apex(t, curve, *APEX_WINDOW)
    if meas_ap and mod_ap:
        row["apex1_time_err_s"] = mod_ap[0] - meas_ap[0]
        row["apex1_height_err_m"] = mod_ap[1] - meas_ap[1]
    else:
        row["apex1_time_err_s"] = row["apex1_height_err_m"] = None
    return row


def plot_dataset(ds, curves, path):
    t, h, m = ds["t"], ds["h"], ds["mask"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    ax.plot(t, h, color="0.75", lw=1, label="measured, gaps interpolated")
    ax.plot(t[m], h[m], ".", color="black", ms=3, label="measured (real readings)")
    for key, c in curves.items():
        ax.plot(t, c, color=COLORS[key], lw=1.6, label=LABELS[key])
        axr.plot(t[m], (c - h)[m], ".", color=COLORS[key], ms=3, label=LABELS[key])
    for a in (ax, axr):
        a.axvline(SEED_SECONDS, color="0.5", ls=":", lw=1)
        a.grid(alpha=0.25)
    ax.text(SEED_SECONDS + 0.01, 0.02, "scoring starts", fontsize=8, color="0.4",
            transform=ax.get_xaxis_transform())
    ax.set_ylabel("height of top of ball (m)")
    ax.set_title(f"{ds['name']}: measured vs models")
    ax.legend(fontsize=8, loc="upper right")
    axr.axhline(0, color="black", lw=0.8)
    axr.set_ylabel("model minus measured (m)")
    axr.set_xlabel("time after drop (s)")
    axr.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(rows, path):
    models = [m for m in ("physics", "lstm", "constant") if any(r["model"] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(models)
    for j, mod in enumerate(models):
        vals = [next(r["rmse_m"] for r in rows if r["model"] == mod and r["dataset"] == d)
                for d in DATASETS]
        xs = np.arange(len(DATASETS)) + j * w
        ax.bar(xs, vals, w, label=LABELS[mod], color=COLORS[mod])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.003, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(DATASETS)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("RMSE (m), t >= 0.2 s")
    ax.set_title("RMSE per dataset")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmts(v, digits=3):
    """Signed number or n/a."""
    return "n/a" if v is None else f"{v:+.{digits}f}"


def write_outputs(rows, cross, fits, lstm_note):
    os.makedirs(DOCS, exist_ok=True)
    cols = ["dataset", "model", "n_scored", "rmse_m", "mae_m",
            "fall_to_20cm_err_s", "apex1_time_err_s", "apex1_height_err_m"]
    with open(os.path.join(DOCS, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else
                            (round(r[k], 5) if isinstance(r[k], float) else r[k]))
                        for k in cols})
    with open(os.path.join(DOCS, "physics_cross_fit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fitted_on", "restitution", "decay", "evaluated_on", "rmse_m", "mae_m"])
        for c in cross:
            w.writerow([c["fit"], c["e"], c["k"], c["on"],
                        round(c["rmse"], 5), round(c["mae"], 5)])

    lines = ["# Evaluation results", "",
             f"Scored where the sensor returned a real reading and t >= {SEED_SECONDS} s. "
             "Heights are the height of the top of the ball in metres. "
             "Timing errors are model minus measured.", "",
             "| Dataset | Model | Points | RMSE (m) | MAE (m) | Fall to 20 cm (s) | "
             "First apex time (s) | First apex height (m) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {LABELS[r['model']]} | {r['n_scored']} | "
                     f"{r['rmse_m']:.3f} | {r['mae_m']:.3f} | {fmts(r['fall_to_20cm_err_s'])} | "
                     f"{fmts(r['apex1_time_err_s'])} | {fmts(r['apex1_height_err_m'])} |")
    fp = fits[FIT_DATASET]
    lines += ["", f"Physics parameters fitted on {FIT_DATASET}: restitution {fp['restitution']}, "
              f"per-bounce decay {fp['decay']} (rubber preset: 59 g, 3 cm radius, Cd 0.47). "
              "Initial downward speed estimated from the first 0.2 s of each dataset: "
              + ", ".join(f"{d} {fits[d]['v0']:.2f} m/s" for d in DATASETS) + ".", "",
              "## Physics cross-fit (RMSE in m; rows = fitted on, columns = evaluated on)", "",
              "| Fitted on (e, decay) | " + " | ".join(DATASETS) + " |",
              "|---|" + "---|" * len(DATASETS)]
    for fit in DATASETS:
        cells = [f"{next(c['rmse'] for c in cross if c['fit'] == fit and c['on'] == on):.3f}"
                 for on in DATASETS]
        lines.append(f"| {fit} ({fits[fit]['restitution']}, {fits[fit]['decay']}) | "
                     + " | ".join(cells) + " |")
    lines += ["", f"LSTM: {lstm_note}", ""]
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Compare physics and LSTM with measured data.")
    ap.add_argument("--skip-lstm", action="store_true", help="evaluate the physics model only")
    ap.add_argument("--epochs", type=int, default=config.TRAINING_EPOCHS,
                    help="max LSTM training epochs")
    args = ap.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    data = {d: load_measured(d) for d in DATASETS}
    for d in DATASETS:
        print(f"{d}: {len(data[d]['t'])} samples, {data[d]['mask'].sum()} backed by real "
              f"readings, estimated initial speed {data[d]['v0']:.2f} m/s")

    print("Fitting physics on each dataset (grid search)...", flush=True)
    fits = {d: fit_physics(data[d]) for d in DATASETS}
    for d, f in fits.items():
        print(f"  fit on {d}: {f}")

    cross = []
    for fit in DATASETS:
        for on in DATASETS:
            ds = data[on]
            c = physics_curve(ds["t"], fits[fit]["restitution"], fits[fit]["decay"], ds["h"][0], ds["v0"])
            sel = ds["mask"] & (ds["t"] >= SEED_SECONDS)
            cross.append({"fit": fit, "e": fits[fit]["restitution"], "k": fits[fit]["decay"],
                          "on": on, "rmse": rmse(c[sel], ds["h"][sel]),
                          "mae": mae(c[sel], ds["h"][sel])})

    physics = {d: physics_curve(data[d]["t"], fits[FIT_DATASET]["restitution"],
                                fits[FIT_DATASET]["decay"], data[d]["h"][0], data[d]["v0"])
               for d in DATASETS}

    lstm, lstm_note = None, "skipped (--skip-lstm)"
    if not args.skip_lstm:
        print("LSTM leave-one-out training...", flush=True)
        lstm, lstm_note = lstm_curves([data[d] for d in DATASETS], args.epochs)
        if lstm is None:
            print(lstm_note, file=sys.stderr)
            lstm_note = "not run: " + lstm_note

    rows = []
    for d in DATASETS:
        ds = data[d]
        sel = ds["mask"] & (ds["t"] >= SEED_SECONDS)
        const = np.full(len(ds["t"]), ds["h"][sel].mean())
        curves = {"physics": physics[d]}
        if lstm:
            curves["lstm"] = lstm[d]
        for name, c in list(curves.items()) + [("constant", const)]:
            r = {"dataset": d, "model": name}
            r.update(score(ds, c))
            rows.append(r)
        plot_dataset(ds, curves, os.path.join(FIGS, f"comparison_{d.replace('.csv', '')}.png"))

    plot_summary(rows, os.path.join(FIGS, "rmse_summary.png"))
    write_outputs(rows, cross, fits, lstm_note)

    with open(os.path.join(DOCS, "results.md"), encoding="utf-8") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
