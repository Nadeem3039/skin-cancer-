import streamlit as st

st.title("Skin Cancer Detection")
st.write("Upload an image for analysis.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
