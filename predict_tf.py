from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Load model and class names once
try:
    model_path = os.path.join("weights", "keras_Model")  # Updated to SavedModel directory
    model = load_model(model_path, compile=False)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class_names = open(os.path.join("weights", "labels.txt"), "r").readlines()

def predict_image(img_pil):
    size = (224, 224)
    image = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = round(float(prediction[0][index]) * 100, 2)

    if class_name.lower() == "perfect":
        return f"✅ This is a **Good** product. (Confidence: {confidence_score}%)"
    else:
        return f"⚠️ Detected an **Abnormal** product. (Confidence: {confidence_score}%)"
