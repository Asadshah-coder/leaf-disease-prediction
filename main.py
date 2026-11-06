import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# TensorFlow Model Prediction Function
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return predictions  # Return prediction results

# Set page configuration
st.set_page_config(page_title="Plant Disease Recognition System", page_icon="🌱", layout="wide")

# Main Page Layout
st.title("Plant Disease Recognition System 🌱")
st.image("home_page.png.png", use_column_width=True)  # Add your banner image

# Description with section box styling
st.markdown("""
Efficiently identify plant diseases and ensure a healthier harvest. Upload an image of the plant to analyze its health.
""")

# Sections for Home
st.markdown("""
## How It Works:
1. **Upload Images**: Go to the **Disease Recognition** section to upload an image of a plant.
2. **Analysis**: Our model will process the image and identify potential diseases.
3. **Results**: View the prediction results along with disease recommendations.

### Why Choose Us?
- **Accuracy**: Powered by advanced machine learning techniques.
- **User-Friendly**: Simple and intuitive interface for a smooth experience.
- **Fast and Efficient**: Get instant results for quick decision-making.
""")

# Buttons for Page Navigation
col1, col2 = st.columns(2)
with col1:
    if st.button("About This Project"):
        st.session_state['page'] = "About"
with col2:
    if st.button("Go to Disease Recognition Page"):
        st.session_state['page'] = "Disease Recognition"

# Section for About
if st.session_state.get('page') == "About":
    st.header("About This Project")
    st.markdown("""
    This Plant Disease Recognition System uses machine learning algorithms to detect plant diseases from images of plant leaves. It was trained on a dataset with 87K images of healthy and diseased crop leaves, divided into 38 categories. The model analyzes the uploaded image and provides predictions on the possible disease.

    #### Dataset Details:
    - **Train**: 70,295 images
    - **Test**: 33 images
    - **Validation**: 17,572 images

    The project aims to assist farmers and gardeners in quickly identifying plant diseases, helping to improve crop yields and plant health.
    """)

# Section for Disease Recognition
if st.session_state.get('page') == "Disease Recognition":
    st.header("Disease Recognition")
    st.markdown("Upload an image of the plant leaves to analyze for disease detection. The model will provide the prediction results for the uploaded image.")

    # Image upload for prediction (supporting 3 images)
    uploaded_images = st.file_uploader("Choose up to 3 images", type=["jpeg", "jpg", "png"], accept_multiple_files=True)

    if uploaded_images:
        col1, col2, col3 = st.columns(3)  # Three columns to display images side by side

        # Display images
        with col1:
            if uploaded_images:
                st.image(uploaded_images[0], caption="Image 1", use_column_width=True)
        with col2:
            if len(uploaded_images) > 1:
                st.image(uploaded_images[1], caption="Image 2", use_column_width=True)
        with col3:
            if len(uploaded_images) > 2:
                st.image(uploaded_images[2], caption="Image 3", use_column_width=True)

        # Prediction Button
        if st.button("Predict"):
            st.snow()
            st.write("Our Predictions:")

            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
                           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                           'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                           'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                           'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                           'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                           'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                           'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 
                           'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
                           'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                           'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 
                           'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

            prediction_results = []
            for i, image in enumerate(uploaded_images):
                # Model prediction for each image
                predictions = model_prediction(image)
                predicted_class = class_names[np.argmax(predictions)]  # Get the predicted class label
                confidence = np.max(predictions) * 100  # Get the confidence value

                if confidence >= 80:  # Show predictions only if confidence is above 80%
                    prediction_results.append([f"Image {i + 1}", predicted_class, round(confidence, 2)])

            # Display results in table format
            if prediction_results:
                df = pd.DataFrame(prediction_results, columns=["Image", "Disease", "Confidence (%)"])
                st.dataframe(df)
            else:
                st.warning("The model's confidence is too low to make a prediction (below 80%).")
