import tensorflow as tf
from flask import Flask, request, jsonify
from utilities import preprocess_image  # Import preprocess_image function from utilities.py
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("sentiment_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    # Preprocess the incoming image
    image = preprocess_image(request.files['image'])
    # Perform prediction using the trained model
    prediction = model.predict(np.expand_dims(image, axis=0))
    # You may need to post-process the prediction for human-readable results
    # In this example, we assume a simple classification task
    predicted_class = np.argmax(prediction)
    class_labels = ["happy", "sad", "neutral"]  # Modify as needed
    result = {"emotion": class_labels[predicted_class]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
