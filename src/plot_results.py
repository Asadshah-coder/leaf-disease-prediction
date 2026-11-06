"""Plot training curves from a training-history JSON file.

Reads ``results/training_history.json`` (the verified history from the
original 10-epoch run: final train accuracy 98.18%, validation
accuracy 96.06%) and writes two figures:

- ``results/figures/accuracy.png`` — train vs. validation accuracy
- ``results/figures/loss.png`` — train vs. validation loss

Usage::

    python src/plot_results.py
    python src/plot_results.py --history results/training_history.json \\
        --out-dir results/figures

The input JSON must contain the keys ``accuracy``, ``val_accuracy``,
``loss`` and ``val_loss`` (lists of equal length, one value per epoch),
as produced by ``src/train.py``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot accuracy/loss curves from training history."
    )
    parser.add_argument(
        "--history", type=str, default="results/training_history.json",
        help="Training history JSON (default: "
             "results/training_history.json)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/figures",
        help="Where to save the figures (default: results/figures)",
    )
    return parser.parse_args()


def plot_curve(epochs, train_values, val_values, title, ylabel,
               out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_values, color="red", marker="o",
            label="Training")
    ax.plot(epochs, val_values, color="blue", marker="o",
            label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    args = parse_args()
    history_path = Path(args.history)
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Training history not found: {history_path}\n"
            "Run src/train.py first, or check the --history path."
        )
    history = json.loads(history_path.read_text())
    for key in ("accuracy", "val_accuracy", "loss", "val_loss"):
        if key not in history:
            raise ValueError(f"Missing key in {history_path}: {key}")
    n_epochs = len(history["accuracy"])
    epochs = list(range(1, n_epochs + 1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_curve(epochs, history["accuracy"], history["val_accuracy"],
               "Training vs. Validation Accuracy", "Accuracy",
               out_dir / "accuracy.png")
    plot_curve(epochs, history["loss"], history["val_loss"],
               "Training vs. Validation Loss", "Loss",
               out_dir / "loss.png")

    print(f"Epochs plotted: {n_epochs}")
    print(f"Final train accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Final val accuracy:   {history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
