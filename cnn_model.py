"""
cnn_model.py - parking-space classification with the CNN trained in Colab.

How it works
------------
1. The admin uploads a photo of the WHOLE parking lot.
2. The photo is cut into one small crop per parking space (a grid, or the boxes in
   data/parking_spaces.json if that file exists).
3. Every crop is resized to 64x64 and given to the CNN, which outputs
   P(occupied) between 0 and 1.
4. We count crops with P >= 0.5 as occupied.

IMPORTANT - this is an ESTIMATE. The grid has to line up with the real parking bays,
and the CNN makes mistakes. Crops with probability close to 0.5 are counted as
"uncertain" and reported, never hidden.

Training/inference consistency: the CNN saved by the notebook already contains the
"Rescaling(1/255)" layer, so here we only resize and pass raw 0-255 pixel values.
"""

import json
import os

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError

IMG_SIZE = 64                 # must match the notebook (IMG_SIZE)
OCCUPIED_THRESHOLD = 0.5      # P(occupied) >= threshold -> occupied
UNCERTAIN_RANGE = (0.35, 0.65)  # probabilities in this band are flagged as uncertain
MAX_IMAGE_PIXELS = 40_000_000


class CNNError(Exception):
    """Readable error shown to the admin (bad image, missing model, ...)."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_cnn(model_path):
    """Load the trained model. Raises CNNError if the file is missing or unreadable."""
    if not os.path.exists(model_path):
        raise CNNError(f"CNN model file not found: {model_path}. Train it with "
                       f"notebooks/cnn_training.ipynb in Colab and copy cnn_model.keras into models/.")
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise CNNError(f"Could not load the CNN model: {exc}")


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #
def open_image(file_or_bytes):
    """Open an uploaded file safely and return an RGB PIL image."""
    try:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        img = Image.open(file_or_bytes)
        img.verify()                       # detects truncated / corrupt files
        file_or_bytes.seek(0)
        img = Image.open(file_or_bytes)
        img.load()
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise CNNError("This file is not a valid or readable JPG/PNG image.")
    if min(img.size) < 100:
        raise CNNError("Image is too small to contain a parking lot (minimum 100 px).")
    return img


def grid_boxes(width, height, rows, cols, margin=0.02):
    """Equal grid of (x0, y0, x1, y1) pixel boxes, row by row. margin = fraction trimmed on each side."""
    boxes = []
    cw, ch = width / cols, height / rows
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cw, r * ch
            x1, y1 = x0 + cw, y0 + ch
            boxes.append((x0 + margin * cw, y0 + margin * ch, x1 - margin * cw, y1 - margin * ch))
    return boxes


def load_boxes(width, height, rows, cols, margin, roi_file=None):
    """
    Use hand-made boxes from a JSON file when it exists, otherwise a plain grid.
    JSON format: [[x, y, w, h], ...] with values between 0 and 1 (fraction of image size).
    """
    if roi_file and os.path.exists(roi_file):
        try:
            with open(roi_file) as f:
                rois = json.load(f)
            return [(x * width, y * height, (x + w) * width, (y + h) * height) for x, y, w, h in rois]
        except Exception as exc:
            raise CNNError(f"Could not read {roi_file}: {exc}")
    return grid_boxes(width, height, rows, cols, margin)


def preprocess_crops(img, boxes):
    """Crop every parking space and resize - same size as in training. Returns (N, 64, 64, 3) float32."""
    crops = []
    for (x0, y0, x1, y1) in boxes:
        crop = img.crop((int(x0), int(y0), int(max(x1, x0 + 1)), int(max(y1, y0 + 1))))
        crops.append(np.asarray(crop.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype="float32"))
    return np.stack(crops)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def estimate_occupancy(model, img, boxes):
    """
    Returns a dict:
      occupied, vacant, total, occupancy_pct, uncertain, probs (list), source='cnn_estimate'
    """
    if model is None:
        raise CNNError("CNN model is not loaded.")
    if not boxes:
        raise CNNError("No parking-space boxes defined.")
    try:
        batch = preprocess_crops(img, boxes)
        probs = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    except Exception as exc:
        raise CNNError(f"CNN inference failed: {exc}")
    if probs.shape[0] != len(boxes) or not np.all(np.isfinite(probs)):
        raise CNNError("CNN returned invalid output.")

    occupied_flags = probs >= OCCUPIED_THRESHOLD
    lo, hi = UNCERTAIN_RANGE
    uncertain = int(np.sum((probs >= lo) & (probs <= hi)))
    total = len(boxes)
    occupied = int(occupied_flags.sum())
    return {
        "occupied": occupied,
        "vacant": total - occupied,
        "total": total,
        "occupancy_pct": round(100.0 * occupied / total, 1),
        "uncertain": uncertain,
        "probs": probs.tolist(),
        "source": "cnn_estimate",
    }


def draw_overlay(img, boxes, probs=None):
    """Return a copy of the image with the parking-space boxes drawn
    (grey = not classified yet, red = occupied, green = vacant, yellow = uncertain)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    lo, hi = UNCERTAIN_RANGE
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if probs is None:
            colour = (200, 200, 200)
        elif lo <= probs[i] <= hi:
            colour = (255, 200, 0)
        elif probs[i] >= OCCUPIED_THRESHOLD:
            colour = (230, 40, 40)
        else:
            colour = (40, 190, 80)
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=max(2, img.size[0] // 400))
    return out
