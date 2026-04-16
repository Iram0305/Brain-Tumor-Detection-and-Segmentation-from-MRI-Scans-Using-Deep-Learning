import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection using Deep Learning")
st.write("Upload an MRI image to detect whether a brain tumor is present.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_mobilenet_final.h5")

model = load_model()

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (150, 150))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

        if prediction > 0.5:
            st.error(f"🛑 Tumor Detected (Confidence: {prediction*100:.2f}%)")
        else:
            st.success(f"✅ No Tumor Detected (Confidence: {(1-prediction)*100:.2f}%)")

st.warning("⚠️ This tool is for educational purposes only and does not replace professional medical diagnosis.")
