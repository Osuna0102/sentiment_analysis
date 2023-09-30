import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    # Preprocess the image (e.g., resizing, normalization)
    image = cv2.resize(image, target_size)  # Resize to a common size
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image
