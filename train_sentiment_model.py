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
emotion_labels = {'happy': 0, 'sad': 1, 'angry': 2, 'neutral': 3, 'fear': 4, 'surpirse': 5, 'disgust': 6}

# Initialize empty lists to store data and labels
data = []
labels = []

for emotion_dir in os.listdir(dataset_dir):
    emotion_label = emotion_labels.get(emotion_dir, -1)
    if emotion_label == -1:
        continue

    # Loop through images in the current emotion folder
    for image_file in os.listdir(os.path.join(dataset_dir, emotion_dir)):
        image_path = os.path.join(dataset_dir, emotion_dir, image_file)
        # Load and preprocess the image as needed
        # Ensure the images are loaded in grayscale
        try:
            image = load_img(image_path, target_size=(48, 48), color_mode="grayscale")  # Change color_mode to "grayscale"
            image = img_to_array(image) / 255.0
            data.append(image)
            labels.append(emotion_label)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

# Convert data and labels to NumPy arrays

data = np.array(data)
labels = np.array(labels)
x_train, x_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

print("Length of data array:", len(data))
print("Length of labels array:", len(labels))


# Reshape the data to match the input shape of the Conv2D layer
x_train = x_train.reshape(-1, 48, 48, 1)
x_test = x_test.reshape(-1, 48, 48, 1)
#data = data[:12]

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),  # Change input_shape to (48, 48, 1)
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')  # 7 output classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Use 'categorical_crossentropy' for one-hot encoded labels
              metrics=['accuracy'])


#model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
# Train the model using all 12 images
model.fit(data, labels, epochs=10)

# Save the trained model
model.save('sentiment_model.h5')
