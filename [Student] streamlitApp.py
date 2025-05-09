import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

# Configuration
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")

# Load model at startup
MODEL_PATH = os.path.join("weights", "keras_Model.h5")
LABELS_PATH = os.path.join("weights", "labels.txt")

try:
    model = load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.stop()

# UI Setup
st.title("InspectorsAlly")
st.caption("AI-Powered Quality Inspection")

# Image Input
input_method = st.radio("Input Method", ["File Upload", "Camera"], horizontal=True)

img = None
if input_method == "File Upload":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=300)
else:
    cam = st.camera_input("Take a Picture")
    if cam:
        img = Image.open(cam)
        st.image(img, caption="Captured Image", width=300)

# Prediction
if st.button("Analyze"):
    if img:
        try:
            # Preprocess
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            if img_array.shape[-1] == 4:  # Remove alpha if present
                img_array = img_array[..., :3]
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            with st.spinner("Analyzing..."):
                pred = model.predict(img_array)[0]
                confidence = round(float(np.max(pred)) * 100, 2)
                class_name = class_names[np.argmax(pred)]
                
                if class_name.lower() == "perfect":
                    st.success(f"✅ Good Product ({confidence}% confidence)")
                else:
                    st.error(f"⚠️ {class_name} Detected ({confidence}% confidence)")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    else:
        st.warning("Please provide an image first")
