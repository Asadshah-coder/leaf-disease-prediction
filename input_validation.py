"""Input validation for the Plant Disease Recognition System (Phase 1).

Two layers, in order:

1. Quality gates (deterministic, no model needed). They reject uploads
   that are corrupt, too small, blank/near-blank, or extremely blurry,
   and warn about moderate blur. Thresholds were measured on real
   images — see the constants below and experiment E4.
2. Plant-vs-non-plant gate (learned). A small MobileNetV2 binary
   classifier (see ``notebooks/train_plant_gate.ipynb``, experiment
   E4) that answers "is this a plant leaf?". It runs BEFORE the
   disease model: a rejected image never receives a disease
   prediction. The gate weights are hosted on Google Drive like the
   disease model; until they exist, the app runs quality gates only
   and says so honestly (``GATE_DRIVE_FILE_ID`` is empty).

Preprocessing note — the two models normalise differently, and mixing
them up is exactly the bug fixed in the disease model in Nov 2026:
- Disease model: pixels divided by 255 (range [0, 1]).
- Gate (MobileNetV2): ``tf.keras.applications.mobilenet_v2.
  preprocess_input`` (range [-1, 1], expects 0–255 input — do NOT
  divide by 255 first).
"""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Thresholds (measured on real images, 7 Nov 2026 — see E4 notes)
# ---------------------------------------------------------------------------

# Reject images smaller than this in either dimension: too little
# detail for any reliable diagnosis.
MIN_DIMENSION = 64

# Reject (near-)blank images: grayscale pixel standard deviation
# below this means the frame is essentially one flat colour.
BLANK_STD = 5.0

# Blur is the variance of the Laplacian of the grayscale image
# (higher = sharper). Measured on real photos, 7 Nov 2026: sharp
# PlantVillage leaves score 670–5,700; Gaussian blur r=1 scores ~30,
# r=2 scores ~3–5, r=3 scores ~2; a 128px image upscaled to 256px
# scores ~45. Reject only when essentially no edge detail remains;
# warn for anything softer than a normal sharp photo.
BLUR_REJECT = 2.0    # below this: reject as extremely blurry
BLUR_WARN = 150.0    # below this: warn that the photo is blurry

# Gate (leaf) decision threshold on the model's leaf probability.
# Fitted in experiment E4: the HIGHEST threshold that still keeps
# leaf recall >= 99% on the validation set (n=1520) — i.e. the
# strictest gate that still lets real leaves through. At this value:
# leaf recall 99.74%, non-leaf rejection 99.87%, AUC 0.9999.
GATE_THRESHOLD = 0.8615

# Google Drive file ID of the trained gate model (public "anyone with
# the link" download, same pattern as the disease model).
GATE_DRIVE_FILE_ID = "1ec13gxnx5qiXlmIUl2h10_hT_2EKxOS8"

GATE_MODEL_FILENAME = "plant_gate_mobilenetv2.keras"
GATE_IMAGE_SIZE = (128, 128)


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness score: variance of the Laplacian (higher = sharper).

    Implemented with NumPy only (no OpenCV dependency): a 3x3
    Laplacian kernel applied to the grayscale image.
    """
    img = gray.astype(np.float64)
    padded = np.pad(img, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4 * img
    )
    return float(np.var(lap))


def check_image_quality(image_bytes: bytes) -> dict:
    """Run the deterministic quality gates on one uploaded image.

    Returns a dict with:
    - ``ok``: True when the image passes all gates.
    - ``reject``: True when the image must not be diagnosed.
    - ``warn``: True when it passes but quality is poor.
    - ``issues``: human-readable reasons (for the UI).
    - ``metrics``: measured numbers (width, height, std, blur).
    """
    metrics: dict = {}
    issues: list[str] = []
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()  # force decoding: catches truncated/corrupt files
    except Exception:
        return {
            "ok": False, "reject": True, "warn": False,
            "issues": ["This file could not be read — it may be corrupt "
                       "or in an unsupported format."],
            "metrics": metrics,
        }
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    metrics["width"] = width
    metrics["height"] = height
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        issues.append(
            f"This image is only {width}×{height} px — too small for a "
            f"reliable diagnosis. Please upload a photo at least "
            f"{MIN_DIMENSION}×{MIN_DIMENSION} px."
        )
        return {"ok": False, "reject": True, "warn": False,
                "issues": issues, "metrics": metrics}

    gray = np.asarray(image.convert("L"), dtype=np.float64)
    std = float(np.std(gray))
    blur = laplacian_variance(gray)
    metrics["pixel_std"] = round(std, 2)
    metrics["blur_score"] = round(blur, 2)

    if std < BLANK_STD:
        issues.append(
            "This image looks blank or nearly blank (almost no variation "
            "between pixels). Please upload a clear photo of a leaf."
        )
        return {"ok": False, "reject": True, "warn": False,
                "issues": issues, "metrics": metrics}
    if blur < BLUR_REJECT:
        issues.append(
            "This photo is extremely blurry, so no reliable diagnosis is "
            "possible. Please take a sharper, well-lit photo of a single leaf."
        )
        return {"ok": False, "reject": True, "warn": False,
                "issues": issues, "metrics": metrics}
    warn = False
    if blur < BLUR_WARN:
        warn = True
        issues.append(
            "This photo is a bit blurry — the result may be less reliable. "
            "A sharper photo gives a better diagnosis."
        )
    return {"ok": True, "reject": False, "warn": warn,
            "issues": issues, "metrics": metrics}


# ---------------------------------------------------------------------------
# Plant-vs-non-plant gate (learned model)
# ---------------------------------------------------------------------------

def find_gate_file() -> Path | None:
    """Return the local gate model file if present, else None."""
    path = Path(__file__).resolve().parent / GATE_MODEL_FILENAME
    return path if path.is_file() else None


def load_gate_model():
    """Load the trained gate model once and cache it (Streamlit).

    Returns the model, or None when no gate weights exist yet (the app
    then runs quality gates only). Imported lazily so this module also
    works outside Streamlit (e.g. in tests).
    """
    import streamlit as st

    @st.cache_resource(show_spinner="Loading the leaf check...")
    def _load():
        path = find_gate_file()
        if path is None and GATE_DRIVE_FILE_ID:
            try:
                import gdown
            except ImportError:
                return None
            dest = Path(__file__).resolve().parent / GATE_MODEL_FILENAME
            url = ("https://drive.google.com/uc?id="
                   f"{GATE_DRIVE_FILE_ID}")
            with st.spinner("Downloading the leaf check (first run only)..."):
                gdown.download(url, str(dest), quiet=True)
            if dest.is_file() and dest.stat().st_size > 0:
                path = dest
        if path is None:
            return None
        import tensorflow as tf

        return tf.keras.models.load_model(str(path))

    return _load()


def gate_leaf_probability(model, image_file) -> float:
    """Probability (0..1) that the image shows a plant leaf.

    Uses MobileNetV2 preprocessing (0–255 input -> [-1, 1]); do NOT
    divide by 255 first.

    NOTE: the gate was trained with ``image_dataset_from_directory``,
    which sorts classes alphabetically, so leaf=0 and nonleaf=1 — the
    model's sigmoid output is P(nonleaf). The leaf probability is
    therefore 1 - output.
    """
    import tensorflow as tf

    image = tf.keras.preprocessing.image.load_img(
        image_file, target_size=GATE_IMAGE_SIZE
    )
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.array([arr])
    prob_nonleaf = model.predict(arr, verbose=0)[0][0]
    return float(1.0 - prob_nonleaf)


def validate_image(image_bytes: bytes, gate_model=None) -> dict:
    """Full validation of one upload: quality gates, then the leaf gate.

    Returns a dict with:
    - ``decision``: "accept" or "reject".
    - ``reasons``: human-readable explanations for the UI.
    - ``quality``: the quality-gate result dict.
    - ``gate_score``: leaf probability, or None when no gate model
      is available yet.
    - ``gate_active``: whether the learned gate ran.
    """
    quality = check_image_quality(image_bytes)
    if quality["reject"]:
        return {
            "decision": "reject",
            "reasons": quality["issues"],
            "quality": quality,
            "gate_score": None,
            "gate_active": False,
        }
    reasons = list(quality["issues"])
    gate_score = None
    gate_active = False
    if gate_model is not None:
        gate_active = True
        try:
            gate_score = gate_leaf_probability(
                gate_model, BytesIO(image_bytes)
            )
        except Exception:
            gate_score = None
        if gate_score is not None and gate_score < GATE_THRESHOLD:
            return {
                "decision": "reject",
                "reasons": [
                    "This does not look like a plant leaf, so no disease "
                    "diagnosis is possible. Please upload a clear photo "
                    "of a single leaf.",
                ],
                "quality": quality,
                "gate_score": gate_score,
                "gate_active": True,
            }
    return {
        "decision": "accept",
        "reasons": reasons,
        "quality": quality,
        "gate_score": gate_score,
        "gate_active": gate_active,
    }
