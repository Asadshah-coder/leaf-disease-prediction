"""Transfer-learning comparison for the Leaf Disease Prediction project.

Trains three models on the same data pipeline and logs each one's
validation accuracy to ``results/model_comparison.csv``:

(a) custom CNN — the original architecture from ``src/model.py``,
    trained from scratch;
(b) MobileNetV2 — pretrained on ImageNet, base frozen, new 38-class
    head (GlobalAveragePooling + Dropout + Dense 38 softmax);
(c) EfficientNetB0 — pretrained on ImageNet, same style of new head.

Why this matters for a research application: reporting only one model's
accuracy says little. A comparison against standard pretrained
baselines shows whether the custom CNN is actually competitive, and it
is the kind of experiment a scholarship reviewer expects to see.

This script is NOT run here — it needs a GPU and TensorFlow. Run it on
Google Colab (see ``docs/COLAB_TRAINING.md``). Use ``--epochs`` to keep
a first comparison short (e.g. 5 epochs), then repeat with more epochs
for the final numbers.

Usage::

    python src/compare_models.py --data-dir data --epochs 5 \\
        --batch-size 32 --results-dir results

    # Re-run one model only, with its own output file:
    python src/compare_models.py --models efficientnetb0 \\
        --output model_comparison_efficientnetb0_rerun.csv

Output: ``results/model_comparison.csv`` with one row per model:
model, val_accuracy, val_loss, epochs, trainable_params.

Preprocessing (fixed after the first E3 run, 8 Nov 2026): the custom
CNN is trained on [0, 1] inputs (its native pipeline), but each
pretrained base gets its own ``preprocess_input`` ([0, 255] -> [-1, 1])
— feeding a frozen ImageNet base plain [0, 1] inputs left
EfficientNetB0 at random chance (val accuracy 0.0273, loss ln 38).
A feature diagnostic runs before each transfer model trains: if the
pooled base features are (near-)constant, the run aborts that model
instead of wasting 5 epochs on a head that cannot learn.
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow running as ``python src/compare_models.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# NOTE: tensorflow is imported lazily inside the functions that need
# it, so that ``python src/compare_models.py --help`` works even on a
# machine without TensorFlow installed.
from model import (CLASS_NAMES, IMAGE_SIZE, build_custom_cnn,
                   compile_model)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare custom CNN vs. pretrained baselines."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Folder containing train/ and valid/ subfolders (default: data)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Epochs per model (default: 5 — keep short for a first run)",
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
        "--results-dir", type=str, default="results",
        help="Where to write model_comparison.csv (default: results)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--models", type=str, default="custom,mobilenetv2,efficientnetb0",
        help="Comma-separated models to train: custom, mobilenetv2, "
             "efficientnetb0 (default: all three)",
    )
    parser.add_argument(
        "--output", type=str, default="model_comparison.csv",
        help="Output CSV filename inside --results-dir "
             "(default: model_comparison.csv)",
    )
    return parser.parse_args()


def make_datasets(data_dir, batch_size, seed, augment=True,
                  preprocess_fn=None):
    """Same pipeline as src/train.py: 128x128, augmentation on the
    training set only.

    ``preprocess_fn`` (e.g. a base's ``preprocess_input``) replaces
    the plain [0, 1] rescaling: images are cast to float32, augmented
    on the [0, 255] scale, then passed through ``preprocess_fn``.
    When None, the original [0, 1] rescaling is used (custom CNN).
    """
    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(Path(data_dir) / "train"), labels="inferred",
        label_mode="categorical", color_mode="rgb",
        batch_size=batch_size, image_size=IMAGE_SIZE,
        shuffle=True, seed=seed,
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        str(Path(data_dir) / "valid"), labels="inferred",
        label_mode="categorical", color_mode="rgb",
        batch_size=batch_size, image_size=IMAGE_SIZE, shuffle=False,
    )
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    def prepare(images, training):
        images = tf.cast(images, tf.float32)
        if training and augment:
            images = augmentation(images, training=True)
        if preprocess_fn is not None:
            return preprocess_fn(images)
        return images / 255.0

    train_ds = train_ds.map(
        lambda x, y: (prepare(x, True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    valid_ds = valid_ds.map(
        lambda x, y: (prepare(x, False), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return (train_ds.prefetch(tf.data.AUTOTUNE),
            valid_ds.prefetch(tf.data.AUTOTUNE))


def transfer_preprocess_fn(base_name):
    """The ImageNet ``preprocess_input`` for a transfer base."""
    import tensorflow as tf

    if base_name == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if base_name == "EfficientNetB0":
        return tf.keras.applications.efficientnet.preprocess_input
    raise ValueError(f"Unknown base model: {base_name}")


def diagnose_base_features(model, valid_ds):
    """Print pooled-feature stats for one validation batch.

    Returns the feature std. A std near zero means the frozen base
    outputs (near-)constant features and the head cannot learn —
    the caller should abort that model instead of training it.
    """
    import numpy as np
    import tensorflow as tf

    base = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
    )
    feats = tf.keras.Model(
        model.input,
        tf.keras.layers.GlobalAveragePooling2D()(base.output),
    )
    images, _ = next(iter(valid_ds))
    out = feats.predict(images, verbose=0)
    std = float(np.std(out))
    print(f"Base features: shape {out.shape}, mean {out.mean():.4f}, "
          f"std {std:.4f}, min {out.min():.4f}, max {out.max():.4f}")
    return std


def build_transfer_model(base_name, input_shape=(128, 128, 3),
                         num_classes=len(CLASS_NAMES)):
    """Pretrained base (frozen) + a new 38-class classification head.

    The caller must feed this model inputs preprocessed with the
    base's own ``preprocess_input`` (see ``transfer_preprocess_fn``):
    feeding plain [0, 1] inputs left EfficientNetB0 at random chance
    in the first E3 run (8 Nov 2026).
    """
    import tensorflow as tf

    if base_name == "MobileNetV2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False,
            weights="imagenet",
        )
    elif base_name == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unknown base model: {base_name}")
    base.trainable = False  # freeze pretrained weights

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def run_one(name, model, train_ds, valid_ds, epochs, learning_rate):
    """Train one model and return its validation metrics."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n=== Training {name} for {epochs} epochs ===")
    model.fit(train_ds, validation_data=valid_ds, epochs=epochs,
              verbose=2)
    loss, accuracy = model.evaluate(valid_ds, verbose=0)
    trainable = sum(
        int(tf.size(w)) for w in model.trainable_weights
    )
    print(f"{name}: val_accuracy={accuracy:.4f} val_loss={loss:.4f}")
    return {
        "model": name,
        "val_accuracy": round(float(accuracy), 4),
        "val_loss": round(float(loss), 4),
        "epochs": epochs,
        "trainable_params": int(trainable),
    }


def main():
    args = parse_args()
    import tensorflow as tf

    tf.random.set_seed(args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    wanted = {m.strip().lower() for m in args.models.split(",")}

    rows = []
    if "custom" in wanted:
        train_ds, valid_ds = make_datasets(
            args.data_dir, args.batch_size, args.seed
        )
        # (a) Custom CNN, trained from scratch.
        custom = compile_model(build_custom_cnn(),
                               learning_rate=args.learning_rate)
        rows.append(run_one("custom_cnn", custom, train_ds, valid_ds,
                            args.epochs, args.learning_rate))

    for base_name, row_name in (
        ("MobileNetV2", "mobilenetv2_imagenet"),
        ("EfficientNetB0", "efficientnetb0_imagenet"),
    ):
        if base_name.lower() not in wanted:
            continue
        # Each transfer base gets its own preprocess_input ([0, 255]
        # -> [-1, 1]); plain [0, 1] inputs broke EfficientNetB0 in
        # the first E3 run.
        train_ds, valid_ds = make_datasets(
            args.data_dir, args.batch_size, args.seed,
            preprocess_fn=transfer_preprocess_fn(base_name),
        )
        model = build_transfer_model(base_name)
        feat_std = diagnose_base_features(model, valid_ds)
        if feat_std < 1e-6:
            print(f"ABORT {row_name}: pooled base features are "
                  f"(near-)constant (std {feat_std:.2e}) — the head "
                  f"cannot learn. Not trained, no row written.")
            continue
        rows.append(run_one(row_name, model, train_ds, valid_ds,
                            args.epochs, args.learning_rate))

    if not rows:
        print("No models trained — nothing written.")
        return
    out_path = results_dir / args.output
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "val_accuracy", "val_loss",
                           "epochs", "trainable_params"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison written to {out_path}")
    for row in rows:
        print(row)
    print("\nNOTE: the pretrained baselines use plain [0, 1] rescaling "
          "here. For final numbers, re-run with each base's own "
          "preprocess_input (see the docstring of "
          "build_transfer_model).")


if __name__ == "__main__":
    main()
