"""Grad-CAM visualisation for the leaf-disease CNN.

What Grad-CAM shows
-------------------
Grad-CAM (Gradient-weighted Class Activation Mapping) answers the
question "which part of the leaf did the model look at when it made
this prediction?" It takes the gradients of the predicted class score
with respect to the feature maps of the last convolutional layer, uses
them to weight those feature maps, and produces a heatmap. Warm areas
(red/yellow) are the regions that most influenced the prediction; cool
areas (blue) mattered little.

For a scholarship/research reviewer this is the explainability evidence:
a model that highlights the diseased spot on the leaf is far more
credible than a model that only reports a high accuracy number.

The last convolutional layer of the custom CNN is the final
``Conv2D(512)`` layer (named automatically by Keras; this script finds
the last ``Conv2D`` layer in the model, so it also works for the
transfer-learning models in ``src/compare_models.py``).

Usage::

    python src/gradcam.py --model trained_plant_disease_model.keras \\
        --image path/to/leaf.jpg --output results/gradcam.png

    python src/gradcam.py --model trained_plant_disease_model.keras \\
        --image path/to/leaf.jpg --output results/gradcam.png \\
        --class-index 5

Options:
    --class-index   class to explain (default: the model's top prediction)
    --alpha         heatmap opacity over the original image (default 0.4)
"""

import argparse
import sys
from pathlib import Path

# Allow running as ``python src/gradcam.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# NOTE: heavy third-party imports (tensorflow, numpy, Pillow,
# matplotlib) are done lazily inside the functions that need them, so
# that ``python src/gradcam.py --help`` works even on a machine
# without TensorFlow installed.
from model import CLASS_NAMES, IMAGE_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grad-CAM heatmap for a leaf-disease prediction."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained .keras model file",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to a leaf image (jpg/png)",
    )
    parser.add_argument(
        "--output", type=str, default="results/gradcam.png",
        help="Where to save the heatmap overlay PNG "
             "(default: results/gradcam.png)",
    )
    parser.add_argument(
        "--class-index", type=int, default=None,
        help="Class index to explain (default: top predicted class)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4,
        help="Heatmap opacity, 0..1 (default: 0.4)",
    )
    return parser.parse_args()


def last_conv_layer(model):
    """Return the name of the last Conv2D layer in the model."""
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(model, image_array, class_index):
    """Compute the Grad-CAM heatmap (values 0..1, size IMAGE_SIZE)."""
    import numpy as np
    import tensorflow as tf

    conv_name = last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    # Global-average-pool the gradients -> one weight per feature map.
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(
        tf.multiply(weights, conv_outputs[0]), axis=-1
    ).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """Blend a heatmap over the original image; returns a PIL image."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm

    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype("uint8")).resize(
            original_image.size, Image.BILINEAR
        )
    ) / 255.0
    coloured = mpl_cm.jet(heatmap_resized)[:, :, :3]  # drop alpha channel
    blended = (1 - alpha) * np.array(original_image) / 255.0 \
        + alpha * coloured
    blended = np.clip(blended, 0, 1)
    plt.close("all")
    return Image.fromarray((blended * 255).astype("uint8"))


def main():
    args = parse_args()
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train one first — see docs/COLAB_TRAINING.md."
        )
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))

    original = Image.open(image_path).convert("RGB")
    image = tf.keras.preprocessing.image.load_img(
        str(image_path), target_size=IMAGE_SIZE
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array / 255.0, axis=0)

    predictions = model.predict(image_array, verbose=0)
    top_index = int(np.argmax(predictions))
    class_index = args.class_index if args.class_index is not None \
        else top_index
    if not 0 <= class_index < len(CLASS_NAMES):
        raise ValueError(
            f"--class-index must be 0..{len(CLASS_NAMES) - 1}"
        )
    confidence = float(predictions[0][top_index])

    print(f"Predicted class: {CLASS_NAMES[top_index]} "
          f"(confidence {confidence:.2%})")
    print(f"Explaining class {class_index}: {CLASS_NAMES[class_index]}")

    heatmap = make_gradcam_heatmap(model, image_array, class_index)
    result = overlay_heatmap(original, heatmap, alpha=args.alpha)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    print(f"Grad-CAM overlay saved to {output_path}")
    print("Warm (red/yellow) areas are the regions that most "
          "influenced this prediction.")


if __name__ == "__main__":
    main()
