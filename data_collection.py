import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from utilities import preprocess_image  # Import preprocess_image function from utilities.py

# Directory containing worker images
dataset_dir = "dataset/"

# Function to load and preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    # Preprocess the image (e.g., resizing, normalization)
    image = cv2.resize(image, target_size)  # Resize to a common size
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Load and preprocess all images
data = []
labels = []

for worker_dir in os.listdir(dataset_dir):
    worker_label = worker_dir  # Worker's label is the directory name
    for image_file in os.listdir(os.path.join(dataset_dir, worker_dir)):
        image_path = os.path.join(dataset_dir, worker_dir, image_file)
        preprocessed_image = preprocess_image(image_path)
        data.append(preprocessed_image)
        labels.append(worker_label)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Save preprocessed data and labels for later use
np.save("x_train.npy", x_train)
np.save("x_test.npy", x_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

x_train = np.load("x_train.npy")
x_test = np.load("x_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")


print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)