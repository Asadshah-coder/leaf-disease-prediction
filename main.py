"""Plant Disease Recognition System — Streamlit app (improved version).

Leaf Disease Prediction Using Deep Learning (Final Year Project).

Changes vs. the original ``main.py``:
- The model is loaded ONCE per session with ``@st.cache_resource`` instead
  of being re-loaded from disk on every single prediction.
- Model filename: tries ``trained_plant_disease_model.keras`` first, then
  falls back to ``trained_model.keras`` (the name the original training
  notebook saved under). If neither file exists, the app shows a clear,
  friendly explanation and how to get the model instead of crashing
  with a bare traceback.
- Fixed deprecated Streamlit arguments: ``use_column_width`` ->
  ``use_container_width``.
- The 80% confidence rule from the original is kept and documented:
  a prediction is only shown when the model's top confidence is >= 80%.

To run the app you need a trained model file next to this script. See
``docs/COLAB_TRAINING.md`` for how to train one on Google Colab, or
``README.md`` for the known limitation in the original project zip.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The model file the app (and the original Test notebook) expects, plus the
# filename the original training notebook actually saved under.
MODEL_CANDIDATES = [
    "trained_plant_disease_model.keras",
    "trained_model.keras",
]

# Class names, exactly as in the original project (order matters: index i of
# the model's softmax output is CLASS_NAMES[i]). Do not reorder.
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

IMAGE_SIZE = (128, 128)

# A prediction is only reported when the model's top confidence is at
# least this high (kept from the original project).
CONFIDENCE_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Model loading (once per session)
# ---------------------------------------------------------------------------

def find_model_file() -> Path | None:
    """Return the first existing model file, or None if neither exists."""
    here = Path(__file__).resolve().parent
    for name in MODEL_CANDIDATES:
        path = here / name
        if path.is_file():
            return path
    return None


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    """Load the trained Keras model once and cache it for the session.

    Returns (model, model_path) or (None, None) when no model file exists.
    """
    path = find_model_file()
    if path is None:
        return None, None
    model = tf.keras.models.load_model(str(path))
    return model, path


def predict_image(model, image_file):
    """Predict the disease class of one uploaded leaf image.

    Returns (predicted_class, confidence) where confidence is 0..1.
    """
    image = tf.keras.preprocessing.image.load_img(
        image_file, target_size=IMAGE_SIZE
    )
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # single image -> batch of one
    predictions = model.predict(input_arr, verbose=0)
    index = int(np.argmax(predictions))
    return CLASS_NAMES[index], float(np.max(predictions))


def show_missing_model_message() -> None:
    """Friendly explanation shown when no trained model file is present."""
    st.error(
        "No trained model file was found, so predictions are unavailable."
    )
    st.markdown(
        """
        The app looks for one of these files next to `main.py`:

        - `trained_plant_disease_model.keras` (preferred name), or
        - `trained_model.keras` (the name the original training notebook
          saved under)

        **How to get a model file:**

        1. Train one on Google Colab — follow the steps in
           `docs/COLAB_TRAINING.md` (run `python src/train.py`, which saves
           the model as `trained_plant_disease_model.keras`), then place
           the downloaded `.keras` file next to `main.py` and restart
           the app; or
        2. If you already have a trained `.keras` file, rename it to
           `trained_plant_disease_model.keras` and place it next to
           `main.py`.

        Everything else in the app (Home and About) works without a model.
        """
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="🌱",
    layout="wide",
)

st.title("Plant Disease Recognition System 🌱")

banner = Path(__file__).resolve().parent / "home_page.png.png"
if banner.is_file():
    st.image(str(banner), use_container_width=True)

st.markdown(
    """
    Efficiently identify plant diseases and ensure a healthier harvest.
    Upload an image of a plant to analyze its health.
    """
)

st.markdown(
    """
    ## How It Works:
    1. **Upload Images**: Go to the **Disease Recognition** section to upload an image of a plant.
    2. **Analysis**: Our model will process the image and identify potential diseases.
    3. **Results**: View the prediction results along with disease recommendations.

    ### Why Choose Us?
    - **Accuracy**: Powered by advanced machine learning techniques.
    - **User-Friendly**: Simple and intuitive interface for a smooth experience.
    - **Fast and Efficient**: Get instant results for quick decision-making.
    """
)

# Load the model once, up front, so we can warn early if it is missing.
model, model_path = load_model()
if model is None:
    show_missing_model_message()
else:
    st.caption(f"Model loaded: `{model_path.name}`")

col1, col2 = st.columns(2)
with col1:
    if st.button("About This Project"):
        st.session_state["page"] = "About"
with col2:
    if st.button("Go to Disease Recognition Page"):
        st.session_state["page"] = "Disease Recognition"

if st.session_state.get("page") == "About":
    st.header("About This Project")
    st.markdown(
        """
        This Plant Disease Recognition System uses machine learning
        algorithms to detect plant diseases from images of plant leaves.
        It was trained on a dataset with 87K images of healthy and
        diseased crop leaves, divided into 38 categories. The model
        analyzes the uploaded image and provides predictions on the
        possible disease.

        #### Dataset Details:
        - **Train**: 70,295 images
        - **Test**: 33 images
        - **Validation**: 17,572 images

        The project aims to assist farmers and gardeners in quickly
        identifying plant diseases, helping to improve crop yields and
        plant health.
        """
    )

if st.session_state.get("page") == "Disease Recognition":
    st.header("Disease Recognition")
    st.markdown(
        "Upload an image of the plant leaves to analyze for disease "
        "detection. The model will provide the prediction results for "
        "the uploaded image."
    )

    if model is None:
        show_missing_model_message()
    else:
        uploaded_images = st.file_uploader(
            "Choose up to 3 images",
            type=["jpeg", "jpg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_images:
            # Show at most 3 images, side by side (as in the original).
            cols = st.columns(3)
            for i, image in enumerate(uploaded_images[:3]):
                with cols[i]:
                    st.image(image, caption=f"Image {i + 1}",
                             use_container_width=True)

            if st.button("Predict"):
                st.snow()
                st.write("Our Predictions:")

                prediction_results = []
                for i, image in enumerate(uploaded_images[:3]):
                    predicted_class, confidence = predict_image(model, image)
                    # 80% confidence rule (kept from the original):
                    # only report a prediction the model is confident about.
                    if confidence >= CONFIDENCE_THRESHOLD:
                        prediction_results.append(
                            [f"Image {i + 1}", predicted_class,
                             round(confidence * 100, 2)]
                        )

                if prediction_results:
                    df = pd.DataFrame(
                        prediction_results,
                        columns=["Image", "Disease", "Confidence (%)"],
                    )
                    st.dataframe(df)
                else:
                    st.warning(
                        "The model's confidence is too low to make a "
                        "prediction (below 80%)."
                    )
