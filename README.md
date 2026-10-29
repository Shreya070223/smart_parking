# Smart Parking Availability Predictor

A college project that combines two deep-learning models in one Streamlit app:

* **CNN** - looks at a parking-lot photo and estimates how many spaces are occupied / vacant.
* **LSTM** - learns from past occupancy and predicts the next 8 half-hour slots (4 hours).

There are exactly two dashboards, both in `app.py`: **User** (status, forecast, booking) and **Admin** (30-minute image upload, prediction-error monitoring, retraining workflow).

> **Honesty notes (put these in your report)**
> * No trained models are included. You train both in Google Colab and copy the files into `models/`.
> * `data/parking_history.csv` is **synthetic** (simulated) until you replace it with real records. A model trained on it learns the simulated pattern only; the app shows a warning when the LSTM config says so.
> * CNN counts are **estimates**, not ground truth. The admin can enter a verified count, which is stored with `source = admin_verified`.
> * Accuracy numbers come from the notebooks' output on *your* run. Do not quote numbers you did not produce.

---

## 1. Architecture

```
 Colab (offline)                         Streamlit app (online)
 ----------------                        ------------------------------------------
 cnn_training.ipynb  --> cnn_model.keras  --> cnn_model.py   (photo -> occupied/vacant)
 lstm_training.ipynb --> lstm_model.keras --> lstm_model.py  (history -> forecast, errors, MAE)
                         scaler.pkl                |
                         lstm_config.json          v
                                              database.py (SQLite: observations, predictions,
                                                           bookings, slots, models, admins)
                                                   |
                                                   v
                                                app.py = User Dashboard + Admin Dashboard
```

```
smart-parking/
├── app.py            both dashboards
├── database.py       SQLite storage, bookings, admin login
├── cnn_model.py      CNN loading + photo -> per-space crops -> counts
├── lstm_model.py     LSTM loading + forecast + error / rolling MAE / retrain flag
├── requirements.txt
├── README.md
├── models/           cnn_model.keras, lstm_model.keras, scaler.pkl, lstm_config.json  (you add these)
├── data/             parking_history.csv (synthetic until replaced), parking.db (created on first run)
└── notebooks/        cnn_training.ipynb, lstm_training.ipynb
```
`lstm_config.json` is the saved "configuration needed for inference" (window, horizon, feature order). `data/parking_spaces.json` is optional (see section 6).

**Flow of one admin upload (5:00 PM slot):** the clock gives the slot (5:00-5:30) -> the database says whether it is already completed -> if not, the uploader is shown and the LSTM prediction for 5:00 is frozen (made from data *before* 5:00) -> photo is cut into spaces -> CNN counts -> one database transaction saves observation + error `|observed vacant - predicted vacant|` + "slot completed" -> rolling MAE is recalculated -> the form disappears. At 5:30 the next slot opens. If CNN fails, nothing is saved and the slot stays open.

---

## 2. Train the models in Google Colab

### CNN (`notebooks/cnn_training.ipynb`)
1. Open https://colab.research.google.com -> File -> Upload notebook -> choose the file. Runtime -> Change runtime type -> **GPU**.
2. Get a per-space dataset (PKLot *segmented* version, or CNRPark-style patches in `Occupied`/`Empty` folders). **Check the dataset's licence and citation rules.** Put it on Drive or paste a direct link (Step 2 of the notebook).
3. Set `DATA_ROOT`, run all cells (Runtime -> Run all). The notebook cleans, splits by day/lot (no leakage), trains, prints accuracy / precision / recall / F1 / confusion matrix and saves + re-loads `cnn_model.keras`.
4. Download `cnn_model.keras` (the last cell downloads it) -> put it in `models/`.

### LSTM (`notebooks/lstm_training.ipynb`)
1. Upload the notebook the same way. A GPU is not needed.
2. Optional: upload your own `parking_history.csv` (columns `timestamp,occupied,capacity`) to `/content`. Without it the notebook generates **labelled synthetic data**.
3. Run all cells. It cleans, resamples to 30 minutes, builds features, splits chronologically 70/15/15, trains, reports MAE/RMSE (against two simple baselines), plots actual vs predicted and saves everything.
4. The last cell downloads `lstm_model.keras`, `scaler.pkl`, `lstm_config.json` (-> `models/`) and the cleaned `parking_history.csv` (-> `data/`, replacing the shipped one).

If Colab disconnects, files in `/content` are lost - download right after training.

---

## 3. Install and run on Windows (VS Code)

Install **Python 3.10-3.12** (tick "Add python.exe to PATH") and VS Code. Open the `smart-parking` folder in VS Code, then Terminal -> New Terminal (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. The app opens at http://localhost:8501. Stop with `Ctrl+C`.

Note: TensorFlow on native Windows is CPU-only, which is fine for inference. Use a Python version supported by your TensorFlow release (3.10-3.12 is safest). Train the model with the same major TensorFlow/Keras version family as the one you install locally (Colab's version is printed at the top of each notebook).

---

## 4. Settings

Edit the top of `app.py` (or set environment variables):

| Setting | Default | Meaning |
|---|---|---|
| `LOT_ROWS` x `LOT_COLS` | 6 x 10 | Grid used to cut the photo into spaces; **capacity = rows x cols** |
| `PARKING_BOOKABLE_FRACTION` | 0.30 | Share of spaces users can reserve in advance |
| `PARKING_MAE_WINDOW` | 12 | Rolling MAE uses the last N completed slots |
| `PARKING_MAE_THRESHOLD` | 3 | MAE (in spaces) above which retraining is flagged |
| `PARKING_MIN_OBS` | 6 | Minimum observations before judging the model |
| `PARKING_TZ` | Asia/Kolkata | Time zone |

Example: `$env:PARKING_MAE_THRESHOLD = 2; streamlit run app.py`

---

## 5. How to test each feature (and demo it)

**First run:** open *Admin Dashboard* -> create the admin account (password is stored only as a salted PBKDF2 hash).

| Feature | How to check |
|---|---|
| Models loaded | Sidebar shows green ticks for CNN and LSTM. Remove a file from `models/` -> red cross + readable message, nothing crashes. |
| User status | Before any upload: "No photo processed yet". After an upload: capacity, occupied, vacant, % and timestamp appear. |
| Forecast | User Dashboard shows a table and chart of the next 8 slots. |
| 30-minute upload | Admin -> section A: shows current slot, uploader, preview with the grid drawn over the photo. Click *Process and submit* -> counts appear, uploader disappears. **Refresh the browser** - it stays closed (state comes from SQLite). |
| Next slot | Wait for :00/:30, or use the sidebar **Demo clock -> Advance 30 minutes** (admin only, for presentations). The uploader returns for the new slot. |
| Missed slot | Skip a slot (advance the clock twice) -> a "Missed slots" warning lists it. |
| Duplicate protection | A slot can be saved only once (database primary key). Try double-clicking Submit. |
| Failed processing | Upload a renamed `.txt` file as `.jpg` -> error, slot stays open. |
| Error monitoring | After uploads: predicted vs observed vacancy, absolute error per slot, rolling MAE, threshold, charts. Example: predicted 20, observed 18 -> error 2. |
| Retrain flag | Set `PARKING_MAE_THRESHOLD=0.5` and `PARKING_MIN_OBS=2`, upload 2-3 slots -> "Retraining: Needed". The old model stays active. |
| Booking | User Dashboard -> pick date, start time (half-hours only), duration (30-minute steps), name, vehicle -> confirmation with reference. |
| No double booking | Book the same vehicle for an overlapping time -> rejected. |
| No overbooking | Set `PARKING_BOOKABLE_FRACTION=0.05` (3 spaces) and book 4 different vehicles for the same slot -> the 4th is rejected. |
| Bookings != occupancy | Bookings change only the "Booked / Bookable left" table, not the occupied count or the forecast. |

**Demo tip:** use a *real* full-lot photo whose bays line up with the 6 x 10 grid (adjust the grid in `app.py` until the overlay matches), or put exact boxes in `data/parking_spaces.json`.

---

## 6. Cropping the photo into spaces

The CNN classifies single spaces, so a full photo must be cut up. Default: an equal grid. For a real lot create `data/parking_spaces.json`:
```json
[[0.05, 0.10, 0.08, 0.12], [0.14, 0.10, 0.08, 0.12]]
```
Each entry is `[x, y, width, height]` as fractions (0-1) of the image. Capacity then equals the number of boxes. (Delete `data/parking.db` if you change capacity after collecting data.)

---

## 7. Manual retraining workflow

1. Observations collect in SQLite automatically.
2. Admin -> section C -> **Download observations CSV**.
3. Upload it to Colab with your old `parking_history.csv` and the old model files; open `lstm_training.ipynb`, set `NEW_OBSERVATIONS_CSV`, `MODEL_VERSION = "v2"`, `CURRENT_MODEL_DIR`.
4. Run all cells - cleaning and retraining happen there.
5. Step 9 evaluates the old and new model on the **same later, unseen data**.
6. Download the new files. 7. Replace them in `models/` **only if the new model is better**, then register the version in section C ("Activate new model"). Rolling MAE restarts for the new version.

*Optional future improvement:* automate this (scheduled job that retrains and promotes a model only if it beats the current one). The app deliberately never runs Colab or retrains by itself.

---

## 8. Known limitations

* Grid-based cropping is approximate; accuracy depends on camera angle and alignment.
* CNN accuracy on the dataset's test days does not transfer automatically to a different camera/lot.
* Prediction error is measured against CNN estimates, so it mixes LSTM and CNN mistakes.
* A missed slot cannot be uploaded later; the forecaster fills gaps with typical values and tells you when it does.
* Login has no rate limiting and the app has one admin role - fine for a college demo, not for production.
