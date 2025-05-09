import streamlit as st
from PIL import Image
from predict_tf import predict_image
import os

# Configuration
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")
MODEL_LOADED = os.path.exists(os.path.join("weights", "keras_Model.h5"))

# UI Components
st.title("InspectorsAlly")
st.caption("Boost Your Quality Control with AI-Powered Inspection")

# Sidebar
with st.sidebar:
    try:
        st.image(Image.open("./docs/overview_dataset.jpg"))
    except:
        st.warning("Dataset overview image not found")
    st.subheader("About")
    st.write("AI-powered quality control inspection system")

# Image Input
input_method = st.radio(
    "Select input method:",
    ["File Upload", "Camera"],
    horizontal=True
)

img = None
if input_method == "File Upload":
    uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", width=300)
else:
    cam = st.camera_input("Take a picture")
    if cam:
        img = Image.open(cam)
        st.image(img, caption="Captured Image", width=300)

# Prediction
if st.button("Analyze Image") and MODEL_LOADED:
    if img:
        with st.spinner("Analyzing..."):
            result = predict_image(img)
            st.subheader("Result")
            st.markdown(result)
    else:
        st.warning("Please provide an image first")
elif not MODEL_LOADED:
    st.error("Model not found. Please ensure keras_Model.h5 exists in weights/ folder")
