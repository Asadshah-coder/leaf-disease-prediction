"""Confidence calibration + OOD rejection analysis (experiment E5).

The disease model is a 38-way softmax classifier. Its raw softmax
outputs are *not* calibrated probabilities: before this analysis the
app treated "100% confidence" as a fact, and its 80% "uncertain" rule
was inherited from the original project without any measurement.

This script measures, on the validation set, and writes
``results/calibration.json``:

1. **Temperature scaling** — fits a single temperature ``T`` on the
   validation logits (minimising negative log-likelihood) and reports
   Expected Calibration Error (ECE) before/after, plus a reliability
   diagram (``results/reliability.png``). Temperature scaling does not
   change the predicted class (argmax is preserved), only the
   confidence values.
2. **OOD scoring** — compares two signals that separate in-distribution
   (validation leaves) from out-of-distribution images (non-leaf
   photos): max-softmax probability (on calibrated outputs) and the
   energy score ``-logsumexp(logits)``. Reports ROC AUC for both,
   picks the better one, and chooses a rejection threshold from the
   ROC curve (the highest threshold that still keeps 95% of real
   leaves above it — the same convention as experiment E4).
3. **Before/after vs the old 80% rule** — how many validation images
   the old rule flagged uncertain, and how many of those were actually
   correct predictions.

The model file itself is never modified. The fitted temperature and
threshold are shipped to the app as constants (see ``main.py``).

Usage::

    python src/calibrate.py --model trained_plant_disease_model.keras \\
        --data-dir data/valid --ood-dir data/ood --batch-size 64

The OOD folder is a flat folder of non-leaf images (no subfolders).
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as ``python src/calibrate.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# NOTE: heavy third-party imports are done lazily inside main(), so
# ``python src/calibrate.py --help`` works without TensorFlow.
from evaluate import check_class_folders
from model import CLASS_NAMES, IMAGE_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate the leaf-disease model and measure OOD rejection."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained .keras model file",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Validation folder with one subfolder per class, e.g. data/valid",
    )
    parser.add_argument(
        "--ood-dir", type=str, required=True,
        help="Flat folder of non-leaf (OOD) images, e.g. data/ood",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for the forward passes (default: 64)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Where to write calibration files (default: results)",
    )
    return parser.parse_args()


def softmax_np(logits):
    """Numerically stable softmax (numpy)."""
    import numpy as np

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def build_logit_model(model, num_classes=len(CLASS_NAMES)):
    """Return a model that outputs pre-softmax logits.

    The saved model's last layer is ``Dense(38, softmax)``. Taking the
    output of the layer before it (Dropout over the 1500-dim features)
    and applying the same Dense weights with a *linear* activation
    gives the exact logits. A self-check in ``main()`` verifies that
    ``softmax(logits)`` matches the original model's output.
    """
    import tensorflow as tf

    features = model.layers[-2].output
    weights, bias = model.layers[-1].get_weights()
    logits = tf.keras.layers.Dense(
        num_classes, activation=None, name="logits"
    )(features)
    logit_model = tf.keras.Model(model.input, logits)
    logit_model.layers[-1].set_weights([weights, bias])
    return logit_model


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE + per-bin reliability data for predicted probabilities."""
    import numpy as np

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            bin_data.append({"bin": [float(lo), float(hi)], "n": 0,
                             "accuracy": None, "confidence": None})
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += (n / len(labels)) * abs(acc - avg_conf)
        bin_data.append({"bin": [float(lo), float(hi)], "n": n,
                         "accuracy": acc, "confidence": avg_conf})
    return float(ece), bin_data


def energy_np(logits):
    """Energy score E(x) = -logsumexp(logits). Lower = more ID-like."""
    import numpy as np

    m = np.max(logits, axis=1)
    return -(m + np.log(np.sum(
        np.exp(logits - m[:, None]), axis=1)))


def fit_temperature(logits, labels):
    """Fit temperature T by minimising NLL of softmax(logits / T).

    Uses scipy when available, otherwise a grid search. Returns
    (T, best NLL).
    """
    import numpy as np

    def nll(t):
        probs = softmax_np(logits / t)
        return float(-np.mean(np.log(np.clip(
            probs[np.arange(len(labels)), labels], 1e-12, 1.0))))

    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        return float(res.x), float(res.fun)
    except ImportError:
        grid = np.arange(0.5, 5.01, 0.05)
        vals = [nll(t) for t in grid]
        best = int(np.argmin(vals))
        return float(grid[best]), float(vals[best])


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, roc_curve

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    ood_dir = Path(args.ood_dir)
    if not ood_dir.is_dir():
        raise FileNotFoundError(f"OOD folder not found: {ood_dir}")
    check_class_folders(data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))
    logit_model = build_logit_model(model)

    # --- Validation logits -------------------------------------------
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(data_dir), labels="inferred", label_mode="int",
        color_mode="rgb", batch_size=args.batch_size,
        image_size=IMAGE_SIZE, shuffle=False,
    )
    normalise = tf.keras.layers.Rescaling(1.0 / 255)
    dataset = dataset.map(
        lambda x, y: (normalise(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    # Self-check: softmax(logits) must match the original model's output.
    first_images, _ = next(iter(dataset))
    orig = model.predict(first_images, verbose=0)
    recon = softmax_np(logit_model.predict(first_images, verbose=0))
    max_diff = float(np.max(np.abs(orig - recon)))
    print(f"Logit-model self-check: max |softmax(logits) - model output| "
          f"= {max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError(
            "Logit extraction does not match the model output "
            f"(max diff {max_diff:.2e}). Refusing to calibrate.")
    print(f"Computing validation logits on {data_dir} ...")
    val_logits, val_labels = [], []
    for i, (images, labels) in enumerate(dataset):
        val_logits.append(logit_model.predict(images, verbose=0))
        val_labels.append(labels.numpy())
        if (i + 1) % 50 == 0:
            print(f"  batch {i + 1} ...")
    val_logits = np.concatenate(val_logits)
    val_labels = np.concatenate(val_labels)
    np.save(results_dir / "val_logits.npy", val_logits)
    np.save(results_dir / "val_labels.npy", val_labels)
    print(f"Validation logits: {val_logits.shape}")

    # --- OOD logits ---------------------------------------------------
    ood_files = sorted(
        [p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
         for p in ood_dir.glob(ext)]
    )
    print(f"OOD images found: {len(ood_files)}")
    if len(ood_files) == 0:
        raise ValueError(f"No images found in {ood_dir}")
    ood_paths = tf.data.Dataset.from_tensor_slices(
        [str(p) for p in ood_files])

    def load_ood(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMAGE_SIZE)
        return tf.cast(img, tf.float32) / 255.0

    ood_ds = ood_paths.map(
        load_ood, num_parallel_calls=tf.data.AUTOTUNE
    ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    print("Computing OOD logits ...")
    ood_logits = np.concatenate(
        [logit_model.predict(b, verbose=0) for b in ood_ds]
    )
    np.save(results_dir / "ood_logits.npy", ood_logits)
    print(f"OOD logits: {ood_logits.shape}")

    # --- Temperature scaling ------------------------------------------
    probs_raw = softmax_np(val_logits)
    temperature, nll_best = fit_temperature(val_logits, val_labels)
    probs_cal = softmax_np(val_logits / temperature)
    acc = float(np.mean(np.argmax(probs_raw, axis=1) == val_labels))
    ece_before, bins_before = expected_calibration_error(
        probs_raw, val_labels)
    ece_after, bins_after = expected_calibration_error(
        probs_cal, val_labels)
    print(f"\nTemperature T = {temperature:.4f} (NLL {nll_best:.4f})")
    print(f"Accuracy (unchanged by calibration): {acc:.4f}")
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after  calibration: {ece_after:.4f}")

    # Reliability diagram (before vs after).
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for bins, label, marker in (
        (bins_before, "Before (raw softmax)", "o"),
        (bins_after, f"After (T = {temperature:.2f})", "s"),
    ):
        xs = [b["confidence"] for b in bins if b["n"] > 0]
        ys = [b["accuracy"] for b in bins if b["n"] > 0]
        ax.plot(xs, ys, marker + "-", label=label)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram (validation set)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(results_dir / "reliability.png", dpi=150)
    plt.close(fig)

    # --- OOD scoring ----------------------------------------------------
    # ID = validation leaves (label 1), OOD = non-leaf images (label 0).
    # Two candidate signals; higher score = more likely in-distribution.
    id_maxprob = np.max(probs_cal, axis=1)
    ood_maxprob = np.max(softmax_np(ood_logits / temperature), axis=1)
    id_energy = energy_np(val_logits)
    ood_energy = energy_np(ood_logits)
    # Energy: lower = more ID-like, so the ROC score is -energy.
    y_true = np.concatenate([np.ones(len(id_maxprob)),
                             np.zeros(len(ood_maxprob))])
    auc_maxprob = float(roc_auc_score(
        y_true, np.concatenate([id_maxprob, ood_maxprob])))
    auc_energy = float(roc_auc_score(
        y_true, np.concatenate([-id_energy, -ood_energy])))
    print(f"\nOOD ROC AUC (max-softmax, calibrated): {auc_maxprob:.4f}")
    print(f"OOD ROC AUC (energy):                   {auc_energy:.4f}")

    if auc_maxprob >= auc_energy:
        signal, id_scores, ood_scores = (
            "max_softmax", id_maxprob, ood_maxprob)
    else:
        signal, id_scores, ood_scores = (
            "energy", -id_energy, -ood_energy)
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, thr = roc_curve(y_true, scores)
    # Highest threshold that still keeps >= 95% of real leaves above it.
    if np.any(tpr >= 0.95):
        idx = int(np.argmax(tpr >= 0.95))
    else:
        idx = int(np.argmax(tpr))
    threshold = float(thr[idx])
    fpr_at_95 = float(fpr[idx])
    id_below = float(np.mean(id_scores < threshold))
    ood_above = float(np.mean(ood_scores >= threshold))
    print(f"Chosen signal: {signal}")
    print(f"OOD threshold (95% TPR): {threshold:.4f}")
    print(f"FPR at 95% TPR: {fpr_at_95:.4f}")
    print(f"ID (leaves) below threshold: {id_below:.4f}")
    print(f"OOD above threshold (missed): {ood_above:.4f}")

    # ROC curve plot.
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, s in (("max-softmax", np.concatenate([id_maxprob, ood_maxprob])),
                    ("energy", np.concatenate([-id_energy, -ood_energy]))):
        f, t, _ = roc_curve(y_true, s)
        ax.plot(f, t, label=f"{name} (AUC {roc_auc_score(y_true, s):.4f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate (OOD called ID)")
    ax.set_ylabel("True positive rate (ID kept)")
    ax.set_title("OOD detection ROC (validation vs non-leaf)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "ood_roc.png", dpi=150)
    plt.close(fig)

    # --- Old 80% rule, for comparison -----------------------------------
    old_flagged = float(np.mean(np.max(probs_raw, axis=1) < 0.80))
    old_flagged_correct = float(np.mean(
        (np.max(probs_raw, axis=1) < 0.80)
        & (np.argmax(probs_raw, axis=1) == val_labels)))
    print(f"\nOld 80% rule: flagged {old_flagged:.4f} of validation as "
          f"uncertain, of which {old_flagged_correct:.4f} were actually "
          f"correct predictions.")

    calibration = {
        "model": str(model_path),
        "n_val": int(len(val_labels)),
        "n_ood": int(len(ood_files)),
        "temperature": temperature,
        "nll_at_temperature": nll_best,
        "accuracy": acc,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "reliability_bins_before": bins_before,
        "reliability_bins_after": bins_after,
        "old_80pct_rule": {
            "fraction_flagged_uncertain": old_flagged,
            "fraction_flagged_but_correct": old_flagged_correct,
        },
        "ood": {
            "signal": signal,
            "auc_max_softmax": auc_maxprob,
            "auc_energy": auc_energy,
            "threshold": threshold,
            "fpr_at_95_tpr": fpr_at_95,
            "id_below_threshold": id_below,
            "ood_above_threshold": ood_above,
        },
    }
    (results_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2)
    )
    print(f"\nFiles written to {results_dir}/: calibration.json, "
          f"reliability.png, ood_roc.png, val_logits.npy, val_labels.npy, "
          f"ood_logits.npy")


if __name__ == "__main__":
    main()
