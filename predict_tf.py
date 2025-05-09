from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Load model at startup
try:
    model = load_model(os.path.join("weights", "keras_Model.h5"), compile=False)
    with open(os.path.join("weights", "labels.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

def predict_image(img_pil):
    """Make prediction with error handling"""
    try:
        # Preprocess image
        size = (224, 224)
        image = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        
        # Normalize and predict
        normalized_image = (image_array.astype(np.float32) / 127.5) - 1
        input_data = np.expand_dims(normalized_image, axis=0)
        
        prediction = model.predict(input_data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence = round(float(prediction[0][index]) * 100, 2)
        
        if class_name.lower() == "perfect":
            return f"✅ This is a **Good** product. (Confidence: {confidence}%)"
        return f"⚠️ Detected **{class_name}**. (Confidence: {confidence}%)"
        
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"
