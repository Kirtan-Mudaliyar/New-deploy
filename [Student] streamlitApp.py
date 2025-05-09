import streamlit as st
import io
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")

st.title("InspectorsAlly")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)

st.write(
    "Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly."
)

with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections. With InspectorsAlly, companies can ensure that their products meet the highest standards of quality, while reducing inspection time and increasing efficiency."
    )

    st.write(
        "This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. InspectorsAlly is capable of identifying even the slightest defects, such as scratches, dents, discolorations, and more on the Leather Product Images."
    )


# Define the functions to load images
def load_uploaded_image(file):
    img = Image.open(file)
    return img


# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio(
    "options", ["File Uploader", "Camera Input"], label_visibility="collapsed"
)

# Check which input method was selected
if input_method == "File Uploader":
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# Load the Keras model
device = "cpu"  # Keras automatically uses the CPU or GPU if available

data_folder = "./data/"
subset_name = "leather"
data_folder = os.path.join(data_folder, subset_name)


def Anomaly_Detection(image_pil, root):
    """
    Given an image and a trained Keras model, returns the predicted class for anomalies.
    """
    model_path = os.path.join("weights", "keras_model.h5")

    # Load the Keras model (inference-only mode)
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()

    model = load_model(model_path, compile=False)  # ✅ Fixed here

    # Preprocess the image to match Keras model input
    image = image_pil.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    
    if image.shape[-1] == 4:
        image = image[..., :3]  # Remove alpha channel if present

    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image)[0]

    if prediction[0] > 0.5:
        return "✅ Your product has been classified as a 'Perfect' item with no anomalies detected."
    else:
        return "⚠️ An anomaly has been detected in your product."


submit = st.button(label="Submit a Product Image")
if submit:
    st.subheader("Output")
    if input_method == "File Uploader":
        img_file_path = uploaded_file_img
    elif input_method == "Camera Input":
        img_file_path = camera_file_img
    prediction = Anomaly_Detection(img_file_path, data_folder)
    with st.spinner(text="This may take a moment..."):
        st.write(prediction)
