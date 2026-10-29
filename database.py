"""
database.py - all SQLite storage for the Smart Parking project.

Tables
------
observations      one row per 30-minute slot (CNN estimate or admin-verified count)
predictions       LSTM prediction for a slot + the error once the slot is observed
completed_slots   which 30-minute slots already have an uploaded image (upload lock)
bookings          user bookings (a logical reservation, NOT physical occupancy)
booking_slots     one row per booked 30-minute slot (used to enforce capacity)
model_versions    which LSTM model is active
retrain_status    single row: is retraining recommended?
admins            admin accounts (salted + hashed passwords)
app_state         tiny key/value store

All slot times are stored as naive local-time strings "YYYY-MM-DD HH:MM:SS"
(Asia/Kolkata by default - the app decides the timezone, this file just stores text).
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

SLOT_MINUTES = 30
TS_FORMAT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- #
# Connection helpers
# --------------------------------------------------------------------------- #
def get_connection(db_path):
    # isolation_level=None -> we control transactions ourselves (BEGIN IMMEDIATE)
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction(db_path):
    """BEGIN IMMEDIATE takes the write lock up front, so two users cannot
    both pass the capacity check and then both insert (no overbooking)."""
    conn = get_connection(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def fmt(dt):
    return dt.strftime(TS_FORMAT)


def parse(text):
    return datetime.strptime(text, TS_FORMAT)


def is_slot_boundary(dt):
    return dt.minute % SLOT_MINUTES == 0 and dt.second == 0 and dt.microsecond == 0


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_start      TEXT NOT NULL UNIQUE,             -- one observation per slot
    submitted_at    TEXT NOT NULL,
    occupied        INTEGER NOT NULL CHECK (occupied >= 0),
    vacant          INTEGER NOT NULL CHECK (vacant >= 0),
    capacity        INTEGER NOT NULL CHECK (capacity > 0),
    occupancy_pct   REAL NOT NULL,
    uncertain       INTEGER NOT NULL DEFAULT 0,       -- spaces the CNN was unsure about
    source          TEXT NOT NULL CHECK (source IN ('cnn_estimate', 'admin_verified')),
    model_version   TEXT,
    CHECK (occupied + vacant = capacity)
);

CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    slot_start       TEXT NOT NULL UNIQUE,            -- the slot that was predicted
    created_at       TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    predicted_vacant INTEGER NOT NULL,
    observed_vacant  INTEGER,                         -- filled when the image arrives
    abs_error        REAL                             -- |observed - predicted|
);

CREATE TABLE IF NOT EXISTS completed_slots (
    slot_start   TEXT PRIMARY KEY,                    -- PRIMARY KEY = no duplicate submissions
    completed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bookings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    reference      TEXT UNIQUE,
    user_name      TEXT NOT NULL,
    vehicle_number TEXT NOT NULL,
    start_slot     TEXT NOT NULL,
    end_slot       TEXT NOT NULL,                     -- exclusive
    n_slots        INTEGER NOT NULL CHECK (n_slots > 0),
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS booking_slots (
    booking_id     INTEGER NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
    slot_start     TEXT NOT NULL,
    vehicle_number TEXT NOT NULL,
    UNIQUE (vehicle_number, slot_start)               -- same vehicle cannot double-book
);
CREATE INDEX IF NOT EXISTS idx_booking_slots_slot ON booking_slots(slot_start);

CREATE TABLE IF NOT EXISTS model_versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    version       TEXT NOT NULL UNIQUE,
    registered_at TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 0,
    test_mae      REAL,                               -- MAE (spaces) measured in Colab on unseen data
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS retrain_status (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    needs_retrain INTEGER NOT NULL DEFAULT 0,
    reason       TEXT,
    rolling_mae  REAL,
    n_obs        INTEGER,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS admins (
    username      TEXT PRIMARY KEY,
    salt          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def init_db(db_path, now, initial_model_version="v1", initial_model_mae=None):
    """Create tables (safe to call on every start)."""
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR IGNORE INTO retrain_status (id, needs_retrain) VALUES (1, 0)")
        conn.execute("INSERT OR IGNORE INTO app_state (key, value) VALUES ('monitoring_started_at', ?)",
                     (fmt(now),))
        has_model = conn.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0]
        if has_model == 0:
            conn.execute(
                "INSERT INTO model_versions (version, registered_at, is_active, test_mae, notes) "
                "VALUES (?, ?, 1, ?, 'Initial model trained in Colab')",
                (initial_model_version, fmt(now), initial_model_mae))
    finally:
        conn.close()


def get_state(db_path, key, default=None):
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Admin authentication (salted PBKDF2 hashes - never plain text)
# --------------------------------------------------------------------------- #
def _hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                               bytes.fromhex(salt_hex), 200_000).hex()


def admin_exists(db_path):
    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM admins").fetchone()[0] > 0
    finally:
        conn.close()


def create_admin(db_path, username, password, now):
    username = username.strip()
    if len(username) < 3:
        raise ValueError("Username must have at least 3 characters.")
    if len(password) < 8:
        raise ValueError("Password must have at least 8 characters.")
    salt = secrets.token_hex(16)
    conn = get_connection(db_path)
    try:
        conn.execute("INSERT INTO admins (username, salt, password_hash, created_at) VALUES (?,?,?,?)",
                     (username, salt, _hash_password(password, salt), fmt(now)))
    except sqlite3.IntegrityError:
        raise ValueError("That username already exists.")
    finally:
        conn.close()


def verify_admin(db_path, username, password):
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT salt, password_hash FROM admins WHERE username = ?",
                           (username.strip(),)).fetchone()
    finally:
        conn.close()
    if row is None:
        _hash_password(password, secrets.token_hex(16))   # keep timing similar
        return False
    return hmac.compare_digest(_hash_password(password, row["salt"]), row["password_hash"])


# --------------------------------------------------------------------------- #
# Observations, upload slots and prediction errors
# --------------------------------------------------------------------------- #
def slot_completed(db_path, slot_start):
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT 1 FROM completed_slots WHERE slot_start = ?",
                           (fmt(slot_start),)).fetchone()
        return row is not None
    finally:
        conn.close()


def get_observation(db_path, slot_start):
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM observations WHERE slot_start = ?",
                           (fmt(slot_start),)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_latest_observation(db_path):
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM observations ORDER BY slot_start DESC LIMIT 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_observed_slots(db_path, start, end):
    """Set of slot_start datetimes that have an observation in [start, end]."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT slot_start FROM observations WHERE slot_start BETWEEN ? AND ?",
                            (fmt(start), fmt(end))).fetchall()
        return {parse(r["slot_start"]) for r in rows}
    finally:
        conn.close()


def save_observation(db_path, slot_start, now, occupied, capacity, uncertain, source,
                     model_version, predicted_vacant):
    """
    Save the observation, the prediction error and the 'slot completed' flag in ONE
    transaction. If anything fails nothing is saved, so the slot stays open.
    Raises ValueError if the slot was already submitted.
    """
    if not is_slot_boundary(slot_start):
        raise ValueError("Slot must start on a 30-minute boundary.")
    if not 0 <= occupied <= capacity:
        raise ValueError("Occupied count must be between 0 and capacity.")
    vacant = capacity - occupied
    pct = round(100.0 * occupied / capacity, 2)
    slot_txt = fmt(slot_start)
    try:
        with transaction(db_path) as conn:
            conn.execute("INSERT INTO completed_slots (slot_start, completed_at) VALUES (?, ?)",
                         (slot_txt, fmt(now)))
            conn.execute(
                "INSERT INTO observations (slot_start, submitted_at, occupied, vacant, capacity, "
                "occupancy_pct, uncertain, source, model_version) VALUES (?,?,?,?,?,?,?,?,?)",
                (slot_txt, fmt(now), occupied, vacant, capacity, pct, uncertain, source, model_version))
            if predicted_vacant is not None:
                # INSERT OR IGNORE keeps an earlier (genuinely 'before the fact') prediction if one exists
                conn.execute(
                    "INSERT OR IGNORE INTO predictions (slot_start, created_at, model_version, predicted_vacant) "
                    "VALUES (?,?,?,?)", (slot_txt, fmt(now), model_version, int(predicted_vacant)))
                conn.execute(
                    "UPDATE predictions SET observed_vacant = ?, abs_error = ABS(? - predicted_vacant) "
                    "WHERE slot_start = ?", (vacant, vacant, slot_txt))
    except sqlite3.IntegrityError:
        raise ValueError("This slot already has a submission.")
    return {"occupied": occupied, "vacant": vacant, "occupancy_pct": pct}


def get_prediction_errors(db_path, model_version=None, limit=None):
    """Prediction rows that already have an observed value, newest first."""
    sql = "SELECT * FROM predictions WHERE abs_error IS NOT NULL"
    params = []
    if model_version:
        sql += " AND model_version = ?"
        params.append(model_version)
    sql += " ORDER BY slot_start DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    conn = get_connection(db_path)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def get_observations(db_path):
    conn = get_connection(db_path)
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM observations ORDER BY slot_start").fetchall()]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Model versions / retraining status
# --------------------------------------------------------------------------- #
def get_active_model(db_path):
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM model_versions WHERE is_active = 1 "
                           "ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_model_versions(db_path):
    conn = get_connection(db_path)
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM model_versions ORDER BY id DESC").fetchall()]
    finally:
        conn.close()


def register_model_version(db_path, version, test_mae, notes, now):
    """Make a newly validated model the active one and clear the retrain flag."""
    version = version.strip()
    if not version:
        raise ValueError("Version name is required.")
    try:
        with transaction(db_path) as conn:
            conn.execute("UPDATE model_versions SET is_active = 0")
            conn.execute("INSERT INTO model_versions (version, registered_at, is_active, test_mae, notes) "
                         "VALUES (?,?,1,?,?)", (version, fmt(now), test_mae, notes))
            conn.execute("UPDATE retrain_status SET needs_retrain = 0, reason = 'New model approved', "
                         "updated_at = ? WHERE id = 1", (fmt(now),))
    except sqlite3.IntegrityError:
        raise ValueError(f"Version '{version}' already exists - choose a new name.")


def set_retrain_status(db_path, needs_retrain, reason, rolling_mae, n_obs, now):
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE retrain_status SET needs_retrain=?, reason=?, rolling_mae=?, n_obs=?, "
                     "updated_at=? WHERE id=1",
                     (int(needs_retrain), reason, rolling_mae, n_obs, fmt(now)))
    finally:
        conn.close()


def get_retrain_status(db_path):
    conn = get_connection(db_path)
    try:
        return dict(conn.execute("SELECT * FROM retrain_status WHERE id = 1").fetchone())
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Bookings
# --------------------------------------------------------------------------- #
def booking_slot_list(start, n_slots):
    return [start + timedelta(minutes=SLOT_MINUTES * i) for i in range(n_slots)]


def create_booking(db_path, user_name, vehicle_number, start, n_slots, bookable_spaces, now,
                   max_slots=8):
    """
    Create a booking after checking (inside one write transaction):
      * start is on a 30-minute boundary and not in the past
      * duration is 1..max_slots half-hour slots
      * this vehicle has no overlapping booking   (UNIQUE constraint + friendly check)
      * every requested slot still has a free bookable space
    Returns the booking as a dict. Raises ValueError with a readable message otherwise.
    """
    user_name = user_name.strip()
    vehicle = "".join(vehicle_number.upper().split())
    if not user_name:
        raise ValueError("Please enter your name.")
    if len(vehicle) < 4:
        raise ValueError("Please enter a valid vehicle number.")
    if not is_slot_boundary(start):
        raise ValueError("Bookings can only start on the hour or half hour (e.g. 9:00, 9:30).")
    if not 1 <= n_slots <= max_slots:
        raise ValueError(f"Duration must be between 30 minutes and {max_slots * SLOT_MINUTES} minutes.")
    current_slot = now.replace(minute=now.minute - now.minute % SLOT_MINUTES, second=0, microsecond=0)
    if start < current_slot:
        raise ValueError("You cannot book a time slot in the past.")

    slots = booking_slot_list(start, n_slots)
    end = slots[-1] + timedelta(minutes=SLOT_MINUTES)
    slot_texts = [fmt(s) for s in slots]
    marks = ",".join("?" * len(slot_texts))

    with transaction(db_path) as conn:
        clash = conn.execute(
            f"SELECT slot_start FROM booking_slots WHERE vehicle_number = ? AND slot_start IN ({marks}) "
            f"ORDER BY slot_start LIMIT 1", [vehicle] + slot_texts).fetchone()
        if clash:
            raise ValueError(f"Vehicle {vehicle} already has a booking at "
                             f"{parse(clash['slot_start']).strftime('%d %b %I:%M %p')} (double booking).")
        full = conn.execute(
            f"SELECT slot_start, COUNT(*) AS n FROM booking_slots WHERE slot_start IN ({marks}) "
            f"GROUP BY slot_start HAVING n >= ? ORDER BY slot_start LIMIT 1",
            slot_texts + [bookable_spaces]).fetchone()
        if full:
            raise ValueError(f"No booking capacity left at "
                             f"{parse(full['slot_start']).strftime('%d %b %I:%M %p')}. Try another time.")
        cur = conn.execute(
            "INSERT INTO bookings (user_name, vehicle_number, start_slot, end_slot, n_slots, created_at) "
            "VALUES (?,?,?,?,?,?)", (user_name, vehicle, slot_texts[0], fmt(end), n_slots, fmt(now)))
        booking_id = cur.lastrowid
        reference = f"PK-{start.strftime('%Y%m%d')}-{booking_id:04d}"
        conn.execute("UPDATE bookings SET reference = ? WHERE id = ?", (reference, booking_id))
        conn.executemany("INSERT INTO booking_slots (booking_id, slot_start, vehicle_number) VALUES (?,?,?)",
                         [(booking_id, t, vehicle) for t in slot_texts])
    return {"reference": reference, "user_name": user_name, "vehicle_number": vehicle,
            "start": start, "end": end, "n_slots": n_slots}


def get_booked_counts(db_path, start, end):
    """{slot_start datetime: number of bookings} for slots in [start, end)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT slot_start, COUNT(*) AS n FROM booking_slots "
                            "WHERE slot_start >= ? AND slot_start < ? GROUP BY slot_start",
                            (fmt(start), fmt(end))).fetchall()
        return {parse(r["slot_start"]): r["n"] for r in rows}
    finally:
        conn.close()


def get_bookings_for_vehicle(db_path, vehicle_number):
    vehicle = "".join(vehicle_number.upper().split())
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT reference, user_name, vehicle_number, start_slot, end_slot, n_slots "
                            "FROM bookings WHERE vehicle_number = ? ORDER BY start_slot DESC LIMIT 20",
                            (vehicle,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def store_prediction(db_path, slot_start, now, model_version, predicted_vacant):
    """Freeze the LSTM prediction for a slot the first time it is computed.
    INSERT OR IGNORE: a later call never overwrites it (keeps the evaluation honest)."""
    conn = get_connection(db_path)
    try:
        conn.execute("INSERT OR IGNORE INTO predictions (slot_start, created_at, model_version, "
                     "predicted_vacant) VALUES (?,?,?,?)",
                     (fmt(slot_start), fmt(now), model_version, int(predicted_vacant)))
        row = conn.execute("SELECT predicted_vacant, model_version FROM predictions WHERE slot_start = ?",
                           (fmt(slot_start),)).fetchone()
        return dict(row)
    finally:
        conn.close()
