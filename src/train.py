"""Reproducible training script for the Leaf Disease Prediction CNN.

Trains the custom CNN from ``src/model.py`` (the exact architecture of
the original training notebook) on the PlantVillage-style dataset and
saves everything a reviewer needs to reproduce the run:

- the trained model, saved with the CORRECT name
  ``trained_plant_disease_model.keras`` (the original notebook saved it
  as ``trained_model.keras`` while ``main.py`` looked for
  ``trained_plant_disease_model.keras`` — that mismatch is fixed here);
- ``results/training_history.json`` — per-epoch accuracy/loss;
- ``results/evaluation.json`` — validation accuracy/loss and per-class
  precision/recall/F1 (from the validation set);
- ``results/classification_report.txt`` — sklearn classification report;
- ``results/confusion_matrix.png`` — confusion matrix figure.

Dataset layout expected under ``--data-dir``::

    <data-dir>/
        train/      # one subfolder per class (38 classes)
        valid/      # one subfolder per class
        test/       # sample images (only 33 in the original zip)

Run on Google Colab with a GPU (see ``docs/COLAB_TRAINING.md``) — full
training on ~70k images is not practical on a CPU-only machine.

Example::

    python src/train.py --data-dir data --epochs 10 --batch-size 32 \\
        --model-out trained_plant_disease_model.keras
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as ``python src/train.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# NOTE: heavy third-party imports (tensorflow, numpy, matplotlib,
# scikit-learn) are done lazily inside the functions that need them,
# so that ``python src/train.py --help`` works even on a machine
# without TensorFlow installed.
from model import CLASS_NAMES, IMAGE_SIZE, build_custom_cnn, compile_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the leaf-disease CNN (custom architecture)."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Folder containing train/ and valid/ subfolders (default: data)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs (default: 10, as in the original run)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.0001,
        help="Adam learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--model-out", type=str, default="trained_plant_disease_model.keras",
        help="Where to save the trained model "
             "(default: trained_plant_disease_model.keras)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Where to write history/evaluation files (default: results)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def make_datasets(data_dir, batch_size, seed):
    """Load train/valid datasets with augmentation on the training set.

    Augmentation (new vs. the original notebook, which had none):
    random horizontal flip, small rotations, and zoom. It is applied
    only to the training set, never to validation.
    """
    import tensorflow as tf

    train_dir = Path(data_dir) / "train"
    valid_dir = Path(data_dir) / "valid"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")
    if not valid_dir.is_dir():
        raise FileNotFoundError(f"Validation folder not found: {valid_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_dir), labels="inferred", label_mode="categorical",
        color_mode="rgb", batch_size=batch_size,
        image_size=IMAGE_SIZE, shuffle=True, seed=seed,
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        str(valid_dir), labels="inferred", label_mode="categorical",
        color_mode="rgb", batch_size=batch_size,
        image_size=IMAGE_SIZE, shuffle=False,
    )

    # Normalise pixel values to [0, 1] and augment the training set only.
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    normalise = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(
        lambda x, y: (augmentation(normalise(x), training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(
        lambda x, y: (normalise(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    return train_ds, valid_ds


def evaluate_on_dataset(model, dataset, results_dir):
    """Evaluate on a dataset; write evaluation.json, classification
    report and confusion-matrix figure. Returns the evaluation dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(np.mean(y_true == y_pred))
    report_text = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4,
        zero_division=0,
    )
    (results_dir / "classification_report.txt").write_text(report_text)

    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4,
        output_dict=True, zero_division=0,
    )
    evaluation = {
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

    # Confusion matrix figure.
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion matrix (validation set)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(CLASS_NAMES, fontsize=6)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return evaluation


def main():
    args = parse_args()
    import numpy as np
    import tensorflow as tf

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    train_ds, valid_ds = make_datasets(
        args.data_dir, args.batch_size, args.seed
    )

    print("Building the custom CNN (see src/model.py) ...")
    model = compile_model(
        build_custom_cnn(), learning_rate=args.learning_rate
    )
    model.summary()

    print(f"Training for {args.epochs} epochs ...")
    history = model.fit(
        train_ds, validation_data=valid_ds, epochs=args.epochs
    )

    # --- Save the model with the CORRECT name -------------------------
    model_path = Path(args.model_out)
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    (results_dir / "training_history.json").write_text(
        json.dumps(history_dict, indent=2)
    )
    print(f"Training history saved to {results_dir / 'training_history.json'}")
    print(f"Final train accuracy: {history_dict['accuracy'][-1]:.4f}")
    print(f"Final val accuracy:   {history_dict['val_accuracy'][-1]:.4f}")

    print("Evaluating on the validation set ...")
    evaluation = evaluate_on_dataset(model, valid_ds, results_dir)
    print(f"Validation accuracy: {evaluation['accuracy']:.4f}")
    print(f"Macro-average F1:    {evaluation['macro_avg_f1']:.4f}")
    print(f"Results written to {results_dir}/")


if __name__ == "__main__":
    main()
