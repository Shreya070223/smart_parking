"""
lstm_model.py - demand forecasting and prediction-error monitoring.

* Loads the LSTM + scaler + config trained in Colab (notebooks/lstm_training.ipynb).
* Builds the model input from history (data/parking_history.csv + new admin observations).
* Predicts occupancy for the next HORIZON half-hour slots.
* Calculates forecast errors, rolling MAE and decides whether retraining is recommended.

The model predicts the OCCUPANCY RATIO (occupied / capacity, between 0 and 1).
Vacant spaces = capacity * (1 - ratio). Using a ratio keeps the model independent of lot size.

FEATURES must have the same names and ORDER as in the notebook (make_features there is
an exact copy of make_features here). load_lstm() refuses to start if they differ.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

import database as db

SLOT = pd.Timedelta(minutes=30)

# Feature order - MUST be identical to the training notebook.
FEATURES = ["occ_ratio", "occ_lag1", "occ_lag48",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]


class LSTMError(Exception):
    """Readable error shown in the dashboard."""


# --------------------------------------------------------------------------- #
# Features (copy of the notebook function)
# --------------------------------------------------------------------------- #
def make_features(ratio):
    """ratio: pandas Series (values 0-1) on a regular 30-minute DatetimeIndex."""
    df = pd.DataFrame({"occ_ratio": ratio.astype("float64")})
    df["occ_lag1"] = df["occ_ratio"].shift(1)       # previous slot
    df["occ_lag48"] = df["occ_ratio"].shift(48)     # same time yesterday
    hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype("float64")
    return df[FEATURES]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_lstm(models_dir):
    """Returns {'model', 'scaler', 'config'}. Raises LSTMError with a clear message if anything is missing."""
    paths = {
        "model": os.path.join(models_dir, "lstm_model.keras"),
        "scaler": os.path.join(models_dir, "scaler.pkl"),
        "config": os.path.join(models_dir, "lstm_config.json"),
    }
    missing = [os.path.basename(p) for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise LSTMError("Missing LSTM file(s) in models/: " + ", ".join(missing) +
                        ". Train with notebooks/lstm_training.ipynb in Colab and copy the files here.")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(paths["model"], compile=False)
        scaler = joblib.load(paths["scaler"])
        with open(paths["config"]) as f:
            config = json.load(f)
    except Exception as exc:
        raise LSTMError(f"Could not load LSTM files: {exc}")
    if config.get("features") != FEATURES:
        raise LSTMError("Feature list in lstm_config.json does not match lstm_model.py. "
                        "Training and inference must use the same features in the same order.")
    return {"model": model, "scaler": scaler, "config": config}


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def load_history(csv_path, observations):
    """
    Build one Series of occupancy ratios on a 30-minute grid.
    csv_path     : data/parking_history.csv (timestamp, occupied, capacity, ...) - may be missing
    observations : rows from database.get_observations() - they override the CSV for the same slot
    (Heavy cleaning is done in Colab; here we only do the light checks needed to be safe.)
    """
    parts = []
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "occupied", "capacity"])
        df = df[(df["capacity"] > 0) & (df["occupied"] >= 0) & (df["occupied"] <= df["capacity"])]
        parts.append(pd.Series((df["occupied"] / df["capacity"]).values,
                               index=df["timestamp"].dt.floor("30min")))
    if observations:
        parts.append(pd.Series([o["occupied"] / o["capacity"] for o in observations],
                               index=pd.to_datetime([o["slot_start"] for o in observations])))
    if not parts:
        return pd.Series(dtype="float64")
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()   # later source (DB) wins
    return s.astype("float64")


def fill_gaps(history, start, end):
    """
    Return (continuous_series, was_real_mask) for [start, end].
    Missing slots are filled with the typical value for that weekday and time
    (falls back to time-of-day average, then the overall mean).
    """
    idx = pd.date_range(start, end, freq="30min")
    s = history.reindex(idx)
    real = s.notna()
    if (~real).any():
        if len(history):
            h = history
            by_dow = h.groupby([h.index.dayofweek, h.index.hour, h.index.minute]).mean()
            by_tod = h.groupby([h.index.hour, h.index.minute]).mean()
            overall = float(h.mean())
        else:
            by_dow, by_tod, overall = pd.Series(dtype=float), pd.Series(dtype=float), 0.5
        filled = []
        for ts, v in s.items():
            if np.isnan(v):
                v = by_dow.get((ts.dayofweek, ts.hour, ts.minute),
                               by_tod.get((ts.hour, ts.minute), overall))
            filled.append(v)
        s = pd.Series(filled, index=idx)
    return s, real


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def predict_ahead(bundle, history, first_slot, capacity):
    """
    Predict the HORIZON slots starting at first_slot, using ONLY data from before first_slot.
    Returns (DataFrame, info) with columns slot_start, pred_ratio, pred_occupied, pred_vacant.
    """
    cfg = bundle["config"]
    window, horizon = int(cfg["window"]), int(cfg["horizon"])
    first_slot = pd.Timestamp(first_slot)
    end = first_slot - SLOT
    start = end - SLOT * (window + 48 - 1)           # extra 48 slots for the lag-48 feature

    past = history[history.index <= end]
    cont, real = fill_gaps(past, start, end)
    feats = make_features(cont).iloc[-window:]
    if feats.isna().any().any():
        raise LSTMError("Not enough history to build the model input.")
    x = bundle["scaler"].transform(feats.values.astype("float64"))
    x = x.reshape(1, window, len(FEATURES)).astype("float32")
    try:
        y = np.asarray(bundle["model"].predict(x, verbose=0)).reshape(-1)[:horizon]
    except Exception as exc:
        raise LSTMError(f"LSTM prediction failed: {exc}")
    ratio = np.clip(y, 0.0, 1.0)
    slots = pd.date_range(first_slot, periods=horizon, freq="30min")
    out = pd.DataFrame({
        "slot_start": slots,
        "pred_ratio": ratio,
        "pred_occupied": np.round(ratio * capacity).astype(int),
    })
    out["pred_vacant"] = capacity - out["pred_occupied"]
    info = {
        "window_real_points": int(real.iloc[-window:].sum()),
        "window_size": window,
        "last_real_observation": (past.index.max().to_pydatetime() if len(past) else None),
    }
    return out, info


# --------------------------------------------------------------------------- #
# Error monitoring
# --------------------------------------------------------------------------- #
def absolute_error(observed_vacant, predicted_vacant):
    """Absolute Error = abs(Actual Vacancy - Predicted Vacancy)"""
    return abs(observed_vacant - predicted_vacant)


def rolling_mae(errors):
    """MAE = sum of absolute errors / number of observations."""
    errors = list(errors)
    return sum(errors) / len(errors) if errors else None


def evaluate_and_flag(db_path, model_version, window, threshold, min_obs, now):
    """
    Called after every successful upload. Uses only the last `window` completed slots predicted
    by the ACTIVE model version. Flags retraining when there are at least `min_obs` observations
    and rolling MAE > threshold. It never retrains anything - the current model stays active.
    """
    rows = db.get_prediction_errors(db_path, model_version=model_version, limit=window)
    n = len(rows)
    mae = rolling_mae(r["abs_error"] for r in rows)
    if n < min_obs:
        needs, reason = False, f"Collecting data: {n}/{min_obs} observations needed before judging the model."
    elif mae > threshold:
        needs, reason = True, f"Rolling MAE {mae:.2f} is above the threshold {threshold:.2f} spaces."
    else:
        needs, reason = False, f"Model OK: rolling MAE {mae:.2f} is within the threshold {threshold:.2f}."
    db.set_retrain_status(db_path, needs, reason, mae, n, now)
    return {"needs_retrain": needs, "reason": reason, "rolling_mae": mae, "n_obs": n}
