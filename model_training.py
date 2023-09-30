import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from utilities import preprocess_image

# Directory containing worker images
dataset_dir = "dataset/"

# Create a label mapping
label_to_int = {
    'happy': 0,
    'sad': 1,
    'angry': 2,
    'neutral': 3
}

# Lists to store data and labels
data = []
labels = []

# Loop through subfolders (emotions) in the dataset directory
for emotion_dir in os.listdir(dataset_dir):
    emotion_label = emotion_dir  # Use the folder name as the emotion label
    emotion_int = label_to_int[emotion_label]  # Convert to integer

    # Loop through images in the current emotion folder
    for image_file in os.listdir(os.path.join(dataset_dir, emotion_dir)):
        image_path = os.path.join(dataset_dir, emotion_dir, image_file)
        preprocessed_image = preprocess_image(image_path)
        data.append(preprocessed_image)
        labels.append(emotion_int)  # Append the integer label

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
