import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

from skimage.color import rgb2gray

# Path to your dataset directory
dataset_dir = "dataset/"

# Define emotions and their corresponding labels
emotion_labels = {'happy': 0, 'sad': 1, 'angry': 2, 'neutral': 3}

# Initialize empty lists to store data and labels
data = []
labels = []

# Loop through subfolders (emotions) in the dataset directory
for emotion_dir in os.listdir(dataset_dir):
    emotion_label = emotion_labels.get(emotion_dir, -1)
    if emotion_label == -1:
        continue

    # Loop through images in the current emotion folder
    for image_file in os.listdir(os.path.join(dataset_dir, emotion_dir)):
        image_path = os.path.join(dataset_dir, emotion_dir, image_file)
        # Load and preprocess the image as needed
        # (you can use the same preprocessing as in data_collection.py)
        try:
            image = load_img(image_path, target_size=(48, 48), color_mode="rgb")
            image = img_to_array(image) / 255.0
            data.append(image)
            labels.append(emotion_label)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")


# Convert data and labels to NumPy arrays

data = np.array(data)
labels = np.array(labels)

# Reshape the data to match the input shape of the Conv2D layer
data = data.reshape(-1, 48, 48, 1)
data = data[:12]

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax') # 4 output classes (happy, sad, angry, neutral)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Use 'categorical_crossentropy' for one-hot encoded labels
              metrics=['accuracy'])

# Train the model using all 12 images
model.fit(data, labels, epochs=10)

# Save the trained model
model.save('sentiment_model.h5')
