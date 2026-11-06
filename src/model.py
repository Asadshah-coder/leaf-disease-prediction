"""Shared model definition for the Leaf Disease Prediction project.

The custom CNN below is an exact reproduction of the architecture from
the original training notebook (``notebooks/Train_plant_disease.ipynb``),
including its one quirk: the second convolution in the 128-filter block
uses 32 filters, not 128. It is kept as-is so results stay comparable
with the original training run (10 epochs, final train accuracy 98.18%,
validation accuracy 96.06% — see ``results/training_history.json``).
"""

# Class names, exactly as in the original project. Order matters: index i
# of the model's softmax output is CLASS_NAMES[i].
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy",
    "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)  # 38
IMAGE_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)


def build_custom_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """Build the original custom CNN (Sequential, Keras).

    Architecture (from the training notebook):
    Conv2D 32 -> Conv2D 32 -> MaxPool
    Conv2D 64 -> Conv2D 64 -> MaxPool
    Conv2D 128 -> Conv2D 32 -> MaxPool
    Conv2D 256 -> Conv2D 256 -> MaxPool
    Conv2D 512 -> Conv2D 512 -> MaxPool
    Dropout(0.25) -> Flatten -> Dense(1500, relu) -> Dropout(0.4)
    -> Dense(num_classes, softmax)

    All convolutions use 3x3 kernels and ReLU; the first uses
    padding='same', the rest use the notebook's settings.
    """
    from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                         MaxPool2D)
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, padding="same",
                     activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding="same",
                     activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding="same",
                     activation="relu"))
    # NOTE: the original notebook uses 32 filters here, not 128.
    # Kept unchanged for comparability with the original run.
    model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=256, kernel_size=3, padding="same",
                     activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=512, kernel_size=3, padding="same",
                     activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=3, activation="relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Dropout(0.25))  # avoid overfitting
    model.add(Flatten())
    model.add(Dense(units=1500, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=num_classes, activation="softmax"))
    return model


def compile_model(model, learning_rate=0.0001):
    """Compile with the original settings: Adam lr 0.0001, categorical
    cross-entropy, accuracy metric."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
