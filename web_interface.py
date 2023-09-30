from flask import Flask, request, render_template
from keras.models import load_model
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained sentiment recognition model
model = load_model('sentiment_model.h5')

# Define emotions corresponding to model output
emotions = ['happy', 'sad', 'angry', 'neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the POST request
        file = request.files['image']

        # Read and preprocess the image
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (48, 48))
        image = image / 255.0
        image = np.reshape(image, (1, 48, 48, 1))

        # Make a prediction
        prediction = model.predict(image)
        predicted_emotion = emotions[np.argmax(prediction)]

        return predicted_emotion

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
