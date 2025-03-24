import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Load the trained model
MODEL_PATH = "skin_cancer_cnn.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error("Model file not found! Please upload 'skin_cancer_cnn.h5' in the working directory.")

# Function to preprocess and predict the image
def predict_skin_cancer(img, model):
    img = img.resize((224, 224))  # Resize image
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model
    prediction = model.predict(img_array)[0]
    return prediction

# Streamlit UI
st.title("🦠 Skin Cancer Detection App")
st.write("Upload an image to check for skin cancer using Deep Learning.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    if st.button("Predict"):  
        prediction = predict_skin_cancer(img, model)
        class_labels = ["Benign", "Malignant"]  # Modify as per your dataset
        predicted_class = class_labels[np.argmax(prediction)]
        
        st.write(f"**Prediction:** {predicted_class}")
        
        # Show prediction confidence
        fig, ax = plt.subplots()
        ax.bar(class_labels, prediction, color=['green', 'red'])
        ax.set_ylabel("Confidence")
        st.pyplot(fig)
pip show tensorflow
