"""
app.py - Smart Parking Availability Predictor (Streamlit).

Exactly two dashboards:
  * User Dashboard  - live occupancy, LSTM forecast, bookings
  * Admin Dashboard - 30-minute image upload (CNN), prediction-error monitoring, retraining workflow

The app NEVER trains or cleans data. It only loads the models trained in Colab.
Run with:   streamlit run app.py
"""

import io
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import cnn_model
import database as db
import lstm_model

# =========================================================================== #
# CONFIGURATION  (change here, or set the environment variable in brackets)
# =========================================================================== #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "parking.db")
HISTORY_CSV = os.path.join(BASE_DIR, "data", "parking_history.csv")
ROI_FILE = os.path.join(BASE_DIR, "data", "parking_spaces.json")      # optional hand-made boxes
MODELS_DIR = os.path.join(BASE_DIR, "models")
CNN_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")

TIMEZONE = os.getenv("PARKING_TZ", "Asia/Kolkata")
LOT_ROWS = int(os.getenv("PARKING_ROWS", 6))          # parking-space grid used to cut the photo
LOT_COLS = int(os.getenv("PARKING_COLS", 10))         #   -> capacity = rows x cols (unless ROI file exists)
GRID_MARGIN = 0.03                                    # trim around every grid cell

BOOKABLE_FRACTION = float(os.getenv("PARKING_BOOKABLE_FRACTION", 0.30))  # share of spaces users may reserve
MAX_BOOKING_SLOTS = 8                                 # 8 x 30 min = 4 hours
BOOKING_DAYS_AHEAD = 7

MAE_WINDOW = int(os.getenv("PARKING_MAE_WINDOW", 12))         # last N completed observations
MAE_THRESHOLD = float(os.getenv("PARKING_MAE_THRESHOLD", 3))  # spaces
MIN_OBSERVATIONS = int(os.getenv("PARKING_MIN_OBS", 6))       # do not judge the model before this many

SLOT_DELTA = timedelta(minutes=30)
TZ = ZoneInfo(TIMEZONE)


def lot_capacity():
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE) as f:
                return len(json.load(f))
        except Exception:
            pass
    return LOT_ROWS * LOT_COLS


CAPACITY = lot_capacity()
BOOKABLE_SPACES = max(1, int(CAPACITY * BOOKABLE_FRACTION))


# =========================================================================== #
# Time helpers  (slot is calculated from the clock - never from session state)
# =========================================================================== #
def get_now():
    """Current local time (naive). The admin 'demo clock' can shift it for presentations."""
    base = datetime.now(TZ).replace(tzinfo=None, microsecond=0)
    return base + timedelta(minutes=st.session_state.get("demo_offset_min", 0))


def floor_slot(dt):
    return dt.replace(minute=dt.minute - dt.minute % 30, second=0, microsecond=0)


def slot_label(dt):
    return dt.strftime("%I:%M %p").lstrip("0")


def full_label(dt):
    return dt.strftime("%d %b %Y, %I:%M %p").replace(" 0", " ")


# =========================================================================== #
# Models (cached - loaded once, never trained here)
# =========================================================================== #
@st.cache_resource(show_spinner="Loading CNN model...")
def get_cnn():
    try:
        return cnn_model.load_cnn(CNN_PATH), None
    except cnn_model.CNNError as exc:
        return None, str(exc)


@st.cache_resource(show_spinner="Loading LSTM model...")
def get_lstm():
    try:
        return lstm_model.load_lstm(MODELS_DIR), None
    except lstm_model.LSTMError as exc:
        return None, str(exc)


def initial_model_info():
    """Version name / MAE written by the Colab notebook (only used the first time the DB is created)."""
    try:
        with open(os.path.join(MODELS_DIR, "lstm_config.json")) as f:
            cfg = json.load(f)
        mae_ratio = cfg.get("test_mae_ratio")
        return cfg.get("version", "v1"), (mae_ratio * CAPACITY if mae_ratio is not None else None)
    except Exception:
        return "v1", None


@st.cache_data(ttl=120, show_spinner=False)
def cached_forecast(slot_iso, model_version, last_obs_slot, capacity):
    """Forecast from `slot` onward. The cache key changes when the slot, model or latest observation changes."""
    bundle, err = get_lstm()
    if bundle is None:
        return None, None, err
    slot = pd.Timestamp(slot_iso)
    observations = db.get_observations(DB_PATH)
    history = lstm_model.load_history(HISTORY_CSV, observations)
    try:
        df, info = lstm_model.predict_ahead(bundle, history, slot, capacity)
        return df, info, None
    except lstm_model.LSTMError as exc:
        return None, None, str(exc)


# =========================================================================== #
# USER DASHBOARD
# =========================================================================== #
def user_dashboard():
    now = get_now()
    slot = floor_slot(now)
    st.title("🅿️ Smart Parking - User Dashboard")
    st.caption(f"Local time ({TIMEZONE}): {full_label(now)}")

    active_model = db.get_active_model(DB_PATH)
    latest = db.get_latest_observation(DB_PATH)

    # ---- current status -------------------------------------------------- #
    st.subheader("Current parking status")
    if latest is None:
        st.info("No parking-lot photo has been processed yet, so live occupancy is not available. "
                "The admin uploads a photo every 30 minutes.")
        c = st.columns(5)
        c[0].metric("Total capacity", CAPACITY)
        for col, name in zip(c[1:], ["Occupied", "Vacant", "Occupancy", "Last observation"]):
            col.metric(name, "-")
    else:
        latest_ts = db.parse(latest["slot_start"])
        c = st.columns(5)
        c[0].metric("Total capacity", latest["capacity"])
        c[1].metric("Occupied now", latest["occupied"])
        c[2].metric("Vacant now", latest["vacant"])
        c[3].metric("Occupancy", f"{latest['occupancy_pct']:.0f}%")
        c[4].metric("Last observation", slot_label(latest_ts), help=full_label(latest_ts))
        st.progress(min(latest["occupancy_pct"] / 100, 1.0))
        kind = "verified by admin" if latest["source"] == "admin_verified" else "estimated by the CNN from a photo"
        st.caption(f"Count {kind}. Camera-based estimates can be off by a few spaces.")
        age_min = (now - latest_ts).total_seconds() / 60
        if age_min > 60:
            st.warning(f"The latest observation is {int(age_min)} minutes old - live numbers may be out of date.")

    # ---- LSTM forecast ---------------------------------------------------- #
    st.subheader("Predicted availability (next 30-minute intervals)")
    forecast, info, err = cached_forecast(slot.isoformat(), active_model["version"],
                                          latest["slot_start"] if latest else "none", CAPACITY)
    if err:
        st.warning(f"Forecast not available: {err}")
    else:
        view = forecast.copy()
        view["Time"] = view["slot_start"].dt.strftime("%I:%M %p")
        view["Predicted occupancy"] = (view["pred_ratio"] * 100).round(0).astype(int).astype(str) + "%"
        left, right = st.columns([3, 2])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=view["slot_start"], y=view["pred_vacant"], mode="lines+markers",
                                     name="Predicted vacant spaces", line=dict(width=3)))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis=dict(title="Vacant spaces", range=[0, CAPACITY]),
                              xaxis=dict(title="Time", tickformat="%I:%M %p"),
                              title="LSTM forecast of vacant spaces")
            st.plotly_chart(fig, width="stretch")
        with right:
            st.dataframe(view[["Time", "pred_occupied", "pred_vacant", "Predicted occupancy"]].rename(
                columns={"pred_occupied": "Occupied", "pred_vacant": "Vacant"}),
                hide_index=True, width="stretch")
        st.caption(f"Forecast by LSTM model '{active_model['version']}'. It is a prediction, not a guarantee.")
        lstm_bundle, _ = get_lstm()
        if lstm_bundle and lstm_bundle["config"].get("trained_on_synthetic_data"):
            st.warning("Demo notice: this LSTM was trained on SYNTHETIC (simulated) data, not on real parking records.")
        if info and info["window_real_points"] < info["window_size"]:
            st.caption(f"Data note: {info['window_size'] - info['window_real_points']} of the last "
                       f"{info['window_size']} input slots had no real reading and were filled with the typical "
                       f"value for that time, so this forecast is less reliable.")

    # ---- booking ---------------------------------------------------------- #
    st.divider()
    booking_section(now, slot, forecast if not err else None)


def booking_section(now, current_slot, forecast):
    st.subheader("Book a parking space")
    st.caption(f"{BOOKABLE_SPACES} of {CAPACITY} spaces can be reserved in advance. Bookings are reservations - "
               f"they are tracked separately from the cars actually in the lot and from the forecast.")

    today = now.date()
    c1, c2, c3 = st.columns(3)
    day = c1.date_input("Date", value=today, min_value=today,
                        max_value=today + timedelta(days=BOOKING_DAYS_AHEAD), key="book_date")
    # Only half-hour boundaries can be chosen, so 9:15 AM is impossible by design
    # (database.create_booking validates this again on the server side).
    day_start = datetime.combine(day, datetime.min.time())
    options = [day_start + SLOT_DELTA * i for i in range(48)]
    options = [o for o in options if o >= current_slot]
    if not options:
        st.info("No time slots left on this date.")
        return
    labels = {slot_label(o): o for o in options}
    start_label = c2.selectbox("Start time", list(labels.keys()), key=f"book_start_{day}")
    duration_opts = {f"{30 * i} min" if i < 2 else f"{(30 * i) // 60}h {(30 * i) % 60:02d}m".replace(" 00m", ""): i
                     for i in range(1, MAX_BOOKING_SLOTS + 1)}
    dur_label = c3.selectbox("Duration", list(duration_opts.keys()), key="book_dur")
    start = labels[start_label]
    n_slots = duration_opts[dur_label]
    end = start + SLOT_DELTA * n_slots

    # live availability for the chosen window
    booked = db.get_booked_counts(DB_PATH, start, end)
    rows = []
    for i in range(n_slots):
        t = start + SLOT_DELTA * i
        pred = None
        if forecast is not None:
            match = forecast.loc[forecast["slot_start"] == pd.Timestamp(t), "pred_vacant"]
            pred = int(match.iloc[0]) if len(match) else None
        rows.append({"Slot": slot_label(t), "Booked": booked.get(t, 0),
                     "Bookable spaces left": BOOKABLE_SPACES - booked.get(t, 0),
                     "Forecast vacant (info)": pred if pred is not None else "-"})
    st.caption(f"Your booking: {full_label(start)} to {slot_label(end)}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    n1, n2 = st.columns(2)
    name = n1.text_input("Your name", key="book_name")
    vehicle = n2.text_input("Vehicle number", key="book_vehicle", placeholder="e.g. MH12AB1234")
    if st.button("Confirm booking", type="primary"):
        try:
            result = db.create_booking(DB_PATH, name, vehicle, start, n_slots, BOOKABLE_SPACES, get_now(),
                                       max_slots=MAX_BOOKING_SLOTS)
            st.session_state["last_booking"] = result
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    last = st.session_state.get("last_booking")
    if last:
        st.success(f"Booking confirmed! Reference **{last['reference']}**  \n"
                   f"{last['user_name']} - {last['vehicle_number']}  \n"
                   f"{full_label(last['start'])} to {slot_label(last['end'])} ({last['n_slots'] * 30} minutes)")

    with st.expander("Look up my bookings"):
        q = st.text_input("Vehicle number", key="lookup_vehicle")
        if q.strip():
            found = db.get_bookings_for_vehicle(DB_PATH, q)
            if found:
                df = pd.DataFrame(found)
                df["start_slot"] = pd.to_datetime(df["start_slot"]).dt.strftime("%d %b %I:%M %p")
                df["end_slot"] = pd.to_datetime(df["end_slot"]).dt.strftime("%d %b %I:%M %p")
                st.dataframe(df[["reference", "user_name", "start_slot", "end_slot"]], hide_index=True,
                             width="stretch")
            else:
                st.info("No bookings found for that vehicle number.")


# =========================================================================== #
# ADMIN DASHBOARD
# =========================================================================== #
def admin_dashboard():
    st.title("🛠️ Smart Parking - Admin Dashboard")
    if not db.admin_exists(DB_PATH):
        st.info("First run: create the admin account.")
        with st.form("create_admin"):
            u = st.text_input("Choose a username")
            p1 = st.text_input("Choose a password (min 8 characters)", type="password")
            p2 = st.text_input("Repeat password", type="password")
            if st.form_submit_button("Create admin"):
                if p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        db.create_admin(DB_PATH, u, p1, get_now())
                        st.success("Admin created. Please log in.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        return

    if not st.session_state.get("admin_user"):
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                if db.verify_admin(DB_PATH, u, p):
                    st.session_state["admin_user"] = u.strip()
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
        return

    top = st.columns([6, 1])
    top[0].caption(f"Logged in as **{st.session_state['admin_user']}**")
    if top[1].button("Log out"):
        st.session_state.pop("admin_user", None)
        st.session_state.pop("demo_offset_min", None)
        st.rerun()

    with st.sidebar.expander("Demo clock (presentation only)"):
        st.caption("Moves this session's clock forward so you can show the next 30-minute slot without "
                   "waiting. Data saved in demo mode gets the shifted timestamp.")
        if st.button("Advance 30 minutes"):
            st.session_state["demo_offset_min"] = st.session_state.get("demo_offset_min", 0) + 30
            st.rerun()
        if st.button("Reset to real time"):
            st.session_state["demo_offset_min"] = 0
            st.rerun()
        if st.session_state.get("demo_offset_min"):
            st.warning(f"Clock shifted by +{st.session_state['demo_offset_min']} min")

    section_upload()
    st.divider()
    section_monitoring()
    st.divider()
    section_retraining()


# ---- A. 30-minute upload --------------------------------------------------- #
def missed_slots(now, slot):
    """Past slots (max 8 = last 4 hours) that never got an image, counted only since monitoring began."""
    started = db.parse(db.get_state(DB_PATH, "monitoring_started_at"))
    first = floor_slot(started)
    if first < started:
        first += SLOT_DELTA
    first = max(first, slot - SLOT_DELTA * 8)
    wanted = []
    t = first
    while t < slot:
        wanted.append(t)
        t += SLOT_DELTA
    if not wanted:
        return []
    done = db.get_observed_slots(DB_PATH, wanted[0], wanted[-1])
    return [t for t in wanted if t not in done]


def section_upload():
    now = get_now()
    slot = floor_slot(now)
    slot_end = slot + SLOT_DELTA
    active_model = db.get_active_model(DB_PATH)
    st.header("A. Parking image upload (every 30 minutes)")

    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    missed = missed_slots(now, slot)
    if missed:
        st.warning("Missed slots (no image uploaded): " + ", ".join(slot_label(m) for m in missed) +
                   ". They cannot be filled in later; the forecaster fills such gaps with typical values.")

    st.markdown(f"**Current slot:** {slot_label(slot)} - {slot_label(slot_end)}")

    # The decision to show the uploader comes from the database, not from session state.
    if db.slot_completed(DB_PATH, slot):
        obs = db.get_observation(DB_PATH, slot)
        st.success(f"✅ Slot {slot_label(slot)} is complete. The next upload opens at {slot_label(slot_end)}.")
        if obs:
            show_observation(obs)
        return

    st.info(f"⏳ Upload required for slot {slot_label(slot)} (closes at {slot_label(slot_end)}).")

    cnn, cnn_err = get_cnn()
    if cnn is None:
        st.error(cnn_err)
        return

    # Freeze the LSTM prediction for this slot (uses only data from BEFORE the slot).
    predicted_vacant = None
    forecast, _, ferr = cached_forecast(slot.isoformat(), active_model["version"],
                                        (db.get_latest_observation(DB_PATH) or {}).get("slot_start", "none"),
                                        CAPACITY)
    if forecast is not None:
        stored = db.store_prediction(DB_PATH, slot, now, active_model["version"],
                                     int(forecast.iloc[0]["pred_vacant"]))
        predicted_vacant = stored["predicted_vacant"]
        st.write(f"LSTM predicted **{predicted_vacant}** vacant spaces for this slot.")
    else:
        st.warning(f"No LSTM prediction for this slot ({ferr}). The observation can still be saved, "
                   f"but no prediction error will be recorded.")

    file = st.file_uploader("Parking-lot photo (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"],
                            key=f"upload_{slot:%Y%m%d%H%M}")        # new key for every slot
    if file is None:
        return

    try:
        img = cnn_model.open_image(file)
        boxes = cnn_model.load_boxes(img.width, img.height, LOT_ROWS, LOT_COLS, GRID_MARGIN, ROI_FILE)
    except cnn_model.CNNError as exc:
        st.error(str(exc))
        return

    st.image(cnn_model.draw_overlay(img, boxes), caption=f"Preview - {len(boxes)} parking spaces will be "
             f"analysed (grey boxes). If they do not line up with the bays, adjust the grid in app.py.",
             width="stretch")

    verified = st.checkbox("I counted the cars myself (use my count instead of the CNN estimate)",
                           key=f"verified_{slot:%Y%m%d%H%M}")
    verified_occupied = None
    if verified:
        verified_occupied = st.number_input("Verified occupied spaces", 0, CAPACITY, 0, key=f"vc_{slot:%H%M}")

    if st.button("Process and submit", type="primary", key=f"submit_{slot:%Y%m%d%H%M}"):
        try:
            if floor_slot(get_now()) != slot:
                raise ValueError("The 30-minute slot changed while you were uploading. Refresh and upload again.")
            with st.spinner("Running CNN..."):
                result = cnn_model.estimate_occupancy(cnn, img, boxes)
            if result["total"] != CAPACITY:
                raise ValueError(f"The photo was split into {result['total']} spaces but capacity is {CAPACITY}.")
            occupied, source = result["occupied"], "cnn_estimate"
            if verified:
                occupied, source = int(verified_occupied), "admin_verified"
            db.save_observation(DB_PATH, slot, get_now(), occupied, CAPACITY, result["uncertain"], source,
                                active_model["version"], predicted_vacant)
            outcome = lstm_model.evaluate_and_flag(DB_PATH, active_model["version"], MAE_WINDOW,
                                                   MAE_THRESHOLD, MIN_OBSERVATIONS, get_now())
            msg = f"Slot {slot_label(slot)} saved: {occupied} occupied, {CAPACITY - occupied} vacant."
            if outcome["needs_retrain"]:
                msg += " ⚠️ Model flagged for retraining."
            st.session_state["flash"] = msg
            st.rerun()        # the form disappears because the DB now says the slot is complete
        except (cnn_model.CNNError, ValueError) as exc:
            st.error(f"Not saved - the slot is still open. {exc}")


def show_observation(obs):
    c = st.columns(4)
    c[0].metric("Occupied", obs["occupied"])
    c[1].metric("Vacant", obs["vacant"])
    c[2].metric("Occupancy", f"{obs['occupancy_pct']:.0f}%")
    c[3].metric("Uncertain spaces", obs["uncertain"], help="Spaces where the CNN probability was between 35% and 65%.")
    label = "Admin-verified count" if obs["source"] == "admin_verified" else "CNN estimate (not verified ground truth)"
    pred = db.get_prediction_errors(DB_PATH)
    row = next((p for p in pred if p["slot_start"] == obs["slot_start"]), None)
    if row:
        st.write(f"Source: **{label}** - LSTM predicted {row['predicted_vacant']} vacant, observed "
                 f"{row['observed_vacant']}, absolute error **{row['abs_error']:.0f}**.")
    else:
        st.write(f"Source: **{label}** - no LSTM prediction was available for this slot.")


# ---- B. Error monitoring --------------------------------------------------- #
def section_monitoring():
    st.header("B. LSTM prediction-error monitoring")
    active = db.get_active_model(DB_PATH)
    status = db.get_retrain_status(DB_PATH)
    rows = db.get_prediction_errors(DB_PATH)             # all versions, newest first
    mine = [r for r in rows if r["model_version"] == active["version"]][:MAE_WINDOW]
    mae = lstm_model.rolling_mae(r["abs_error"] for r in mine)

    c = st.columns(5)
    c[0].metric("Active model", active["version"])
    c[1].metric(f"Rolling MAE (last {MAE_WINDOW})", "-" if mae is None else f"{mae:.2f} spaces")
    c[2].metric("Error threshold", f"{MAE_THRESHOLD:.1f} spaces")
    c[3].metric("Observations used", f"{len(mine)} / min {MIN_OBSERVATIONS}")
    c[4].metric("Retraining", "⚠️ Needed" if status["needs_retrain"] else "Not needed")
    if status["needs_retrain"]:
        st.error(status["reason"] + " Follow the workflow in section C. The current model stays active "
                 "until a new model is validated and approved.")
    else:
        st.caption(status["reason"] or "No observations yet.")
    st.caption("Observed vacancy comes from the CNN (an estimate) unless the admin entered a verified count, so "
               "errors include CNN mistakes as well as LSTM mistakes.")

    if not rows:
        st.info("No completed predictions yet - upload a photo in section A.")
        return

    df = pd.DataFrame(rows).sort_values("slot_start")
    df["slot_start"] = pd.to_datetime(df["slot_start"])
    df["rolling_mae"] = df["abs_error"].rolling(MAE_WINDOW, min_periods=1).mean()
    src = {o["slot_start"]: o["source"] for o in db.get_observations(DB_PATH)}
    df["Observed from"] = df["slot_start"].dt.strftime(db.TS_FORMAT).map(src)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["slot_start"], y=df["abs_error"], name="Absolute error"))
        fig.add_trace(go.Scatter(x=df["slot_start"], y=df["rolling_mae"], name="Rolling MAE", mode="lines+markers"))
        fig.add_hline(y=MAE_THRESHOLD, line_dash="dash", annotation_text="threshold")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Prediction error history",
                          yaxis_title="Spaces", xaxis_tickformat="%d %b %I:%M %p", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["slot_start"], y=df["predicted_vacant"], name="Predicted vacant (LSTM)",
                                  mode="lines+markers"))
        fig2.add_trace(go.Scatter(x=df["slot_start"], y=df["observed_vacant"], name="Observed vacant",
                                  mode="lines+markers"))
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Predicted vs observed vacancy",
                           yaxis_title="Vacant spaces", xaxis_tickformat="%d %b %I:%M %p",
                           legend=dict(orientation="h"))
        st.plotly_chart(fig2, width="stretch")

    table = df.sort_values("slot_start", ascending=False).head(30).copy()
    table["Slot"] = table["slot_start"].dt.strftime("%d %b %I:%M %p")
    st.dataframe(table[["Slot", "predicted_vacant", "observed_vacant", "abs_error", "Observed from", "model_version"]]
                 .rename(columns={"predicted_vacant": "Predicted vacant", "observed_vacant": "Observed vacant",
                                  "abs_error": "Absolute error", "model_version": "Model"}),
                 hide_index=True, width="stretch")


# ---- C. Retraining workflow ------------------------------------------------ #
def section_retraining():
    st.header("C. Model retraining workflow (manual, in Google Colab)")
    status = db.get_retrain_status(DB_PATH)
    st.markdown(
        "1. **Collect** - every upload is stored in SQLite (automatic).\n"
        "2. **Export** - download the observations CSV below.\n"
        "3. **Upload** the CSV to Colab and open `notebooks/lstm_training.ipynb` (set `NEW_OBSERVATIONS_CSV`).\n"
        "4. **Clean and retrain** - the notebook cleans the data and trains a new LSTM.\n"
        "5. **Evaluate** on later, unseen data - compare the new test MAE with the current model.\n"
        "6. **Download** `lstm_model.keras`, `scaler.pkl`, `lstm_config.json`.\n"
        "7. **Replace** the files in `models/` only if the new model is better, then register it below.")
    st.caption("Optional future improvement: schedule this pipeline automatically (e.g. a Colab/Cloud job that "
               "retrains weekly and only promotes a model that beats the current one). Not part of this version.")

    obs = db.get_observations(DB_PATH)
    export = pd.DataFrame(obs)
    if export.empty:
        csv_bytes = b"timestamp,occupied,capacity,source,uncertain\n"
    else:
        export = export.rename(columns={"slot_start": "timestamp"})[
            ["timestamp", "occupied", "capacity", "source", "uncertain"]]
        csv_bytes = export.to_csv(index=False).encode("utf-8")
    st.download_button(f"⬇️ Download observations CSV ({len(obs)} rows)", csv_bytes,
                       file_name="parking_observations_export.csv", mime="text/csv")
    st.write(f"Retraining status: **{'NEEDED' if status['needs_retrain'] else 'not needed'}** - "
             f"{status['reason'] or 'no data yet'}")

    with st.expander("Register a new, validated model version"):
        st.caption("Do this only AFTER the new files are copied into models/ and the new model beat the old one "
                   "on unseen data in Colab. Until then the current model keeps running.")
        active = db.get_active_model(DB_PATH)
        with st.form("register_model"):
            version = st.text_input("New version name", placeholder="v2")
            new_mae = st.number_input("New test MAE from Colab (spaces)", min_value=0.0, value=0.0, step=0.1)
            notes = st.text_input("Notes", placeholder="Retrained with 3 weeks of CNN observations")
            ok = st.checkbox("I validated the new model on later, unseen data and copied the 3 files into models/")
            if st.form_submit_button("Activate new model"):
                if not ok:
                    st.error("Please confirm validation first.")
                else:
                    try:
                        db.register_model_version(DB_PATH, version, new_mae, notes, get_now())
                        get_lstm.clear()
                        cached_forecast.clear()
                        st.session_state["flash"] = f"Model '{version}' is now active."
                        if active and active["test_mae"] is not None and new_mae >= active["test_mae"]:
                            st.session_state["flash"] += " Note: its MAE is not better than the previous model."
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    versions = pd.DataFrame(db.list_model_versions(DB_PATH))
    st.dataframe(versions[["version", "registered_at", "is_active", "test_mae", "notes"]], hide_index=True,
                 width="stretch")


# =========================================================================== #
# MAIN
# =========================================================================== #
@st.fragment(run_every="15s")
def slot_watcher(rendered_slot):
    """Invisible helper: when a new 30-minute slot starts, re-run the whole page
    (so the next upload form appears and the forecast refreshes)."""
    if floor_slot(get_now()) != rendered_slot:
        st.rerun()


def main():
    st.set_page_config(page_title="Smart Parking Predictor", page_icon="🅿️", layout="wide")
    version, mae = initial_model_info()
    db.init_db(DB_PATH, get_now(), version, mae)

    st.sidebar.title("Smart Parking")
    page = st.sidebar.radio("Dashboard", ["User Dashboard", "Admin Dashboard"])
    with st.sidebar:
        cnn, cnn_err = get_cnn()
        lstm, lstm_err = get_lstm()
        st.caption("Models")
        st.write(("✅" if cnn else "❌") + " CNN (image analysis)")
        st.write(("✅" if lstm else "❌") + " LSTM (forecast)")
        if st.button("Reload models"):
            get_cnn.clear()
            get_lstm.clear()
            cached_forecast.clear()
            st.rerun()
        if cnn_err or lstm_err:
            st.caption("Train the models in Colab and copy the files into models/ (see README).")

    if page == "User Dashboard":
        user_dashboard()
    else:
        admin_dashboard()
    slot_watcher(floor_slot(get_now()))


main()
