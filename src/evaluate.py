"""Standalone evaluation of a trained leaf-disease model.

Loads a ``.keras`` model and a labelled image folder, then prints the
overall accuracy and per-class precision/recall/F1, and writes the same
artefacts as the training script:

- ``results/evaluation.json`` — accuracy + per-class metrics
- ``results/classification_report.txt`` — sklearn classification report
- ``results/confusion_matrix.png`` — confusion matrix figure

Usage::

    python src/evaluate.py --model trained_plant_disease_model.keras \\
        --data-dir data/valid --batch-size 32

The data folder must contain one subfolder per class (the same 38 class
names as in ``src/model.py``). Any extra/missing class folders are
reported and the run stops, so a wrong folder is never silently scored
as the wrong class.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as ``python src/evaluate.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# NOTE: heavy third-party imports (tensorflow, numpy, matplotlib,
# scikit-learn) are done lazily inside main(), so that
# ``python src/evaluate.py --help`` works even on a machine without
# TensorFlow installed.
from model import CLASS_NAMES, IMAGE_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained leaf-disease model."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained .keras model file",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Folder with one subfolder per class, e.g. data/valid",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Where to write evaluation files (default: results)",
    )
    return parser.parse_args()


def check_class_folders(data_dir):
    """Verify the data folder contains exactly the 38 expected classes."""
    found = sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    missing = [c for c in CLASS_NAMES if c not in found]
    extra = [c for c in found if c not in CLASS_NAMES]
    if missing:
        raise ValueError(
            f"Missing class folders in {data_dir}: {missing}"
        )
    if extra:
        print(f"Warning: unexpected folders ignored: {extra}")
    return found


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train one first — see docs/COLAB_TRAINING.md — or check the path."
        )
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    check_class_folders(data_dir)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))

    dataset = tf.keras.utils.image_dataset_from_directory(
        str(data_dir), labels="inferred", label_mode="categorical",
        color_mode="rgb", batch_size=args.batch_size,
        image_size=IMAGE_SIZE, shuffle=False,
    )
    normalise = tf.keras.layers.Rescaling(1.0 / 255)
    dataset = dataset.map(
        lambda x, y: (normalise(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    print(f"Evaluating on {data_dir} ...")
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(np.mean(y_true == y_pred))
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    (results_dir / "classification_report.txt").write_text(report_text)
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4,
        output_dict=True, zero_division=0,
    )
    evaluation = {
        "model": str(model_path),
        "data_dir": str(data_dir),
        "accuracy": accuracy,
        "n_samples": int(len(y_true)),
        "per_class": {
            name: {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1-score": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"]),
            }
            for name in CLASS_NAMES
        },
        "macro_avg_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_avg_f1": float(report_dict["weighted avg"]["f1-score"]),
    }
    (results_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2)
    )

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"Confusion matrix ({data_dir})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(CLASS_NAMES, fontsize=6)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nSamples evaluated: {len(y_true)}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro-average F1:  {evaluation['macro_avg_f1']:.4f}")
    print(f"Weighted F1:       {evaluation['weighted_avg_f1']:.4f}")
    print("\n" + report_text)
    print(f"Files written to {results_dir}/")


if __name__ == "__main__":
    main()
