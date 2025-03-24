import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Model Load
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")  # Model ka correct naam likho
    return model

model = load_model()

st.title("Skin Cancer Detection")
st.write("Upload an image for analysis.")

# Image Upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Image Processing
    image = image.resize((224, 224))  # Model ke input size ke mutabiq resize karo
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Model Prediction
    prediction = model.predict(image_array)
    class_names = ["Benign", "Malignant"]  # Tumhare model ke labels
    result = class_names[np.argmax(prediction)]

    st.write(f"**Prediction:** {result}")
