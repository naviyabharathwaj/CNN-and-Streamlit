import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model("flower_classifier.keras")

# Flower class names
class_names = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]

# Streamlit page configuration
st.set_page_config(page_title="Flower Species Classifier")

# Title
st.title("Flower Species Classification")
st.write("Upload an image of a flower to predict its species.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png"]
)

# Prediction
if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((180, 180))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    # Display result
    st.subheader("Prediction")
    st.success(f"Flower Species: {predicted_class}")

    st.subheader("Confidence")
    st.write(f"{confidence:.2f}%")