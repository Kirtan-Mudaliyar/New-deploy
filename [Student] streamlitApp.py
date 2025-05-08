import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

# Disable scientific notation
np.set_printoptions(suppress=True)

# Page config
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")
st.title("InspectorsAlly")

st.caption("Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App")
st.write("Try uploading or capturing a product image and watch how an AI model classifies it as 'Good' or 'Anomaly'.")

# Sidebar info
with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections."
    )
    st.write(
        "It uses computer vision and deep learning to identify defects like scratches, dents, and discolorations in leather product images."
    )

# Load image function
def load_uploaded_image(file):
    img = Image.open(file)
    return img

# Input method
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")
elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# Anomaly Detection using Keras model
def Anomaly_Detection(pil_image):
    model_path = "./weights/keras_model.h5"
    threshold = 0.5

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)

    # Preprocess image
    img = pil_image.resize((224, 224))
    img_array = np.asarray(img).astype(np.float32) / 255.0

    # Convert grayscale or RGBA to RGB
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    max_conf = np.max(predictions)

    predicted_class = "Good" if max_conf < threshold else "Anomaly"

    if predicted_class == "Good":
        return "Congratulations! Your product has been classified as a 'Good' item with no anomalies detected."
    else:
        return "We're sorry to inform you that our AI-based inspection system has detected an anomaly in your product."

# Submit button
submit = st.button(label="Submit a Leather Product Image")
if submit:
    st.subheader("Output")
    if input_method == "File Uploader" and uploaded_file:
        img_file = uploaded_file_img
    elif input_method == "Camera Input" and camera_image_file:
        img_file = camera_file_img
    else:
        img_file = None

    if img_file:
        with st.spinner(text="This may take a moment..."):
            prediction = Anomaly_Detection(img_file)
            st.write(prediction)
    else:
        st.warning("No image available for prediction.")
