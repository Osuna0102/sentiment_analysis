from flask import Flask, request, render_template, g
from keras.models import load_model
import cv2
import numpy as np
import face_recognition
import sqlite3

app = Flask(__name__)

# Load the sentiment analysis model
model = load_model('sentiment_model.h5')


# Initialize SQLite connection
conn = sqlite3.connect('database\workers.db')
cursor = conn.cursor()

# Function to get the database connection
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('workers.db')
    return db

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the uploaded image
    image = request.files['image']
    image_path = 'temp.jpg'  # Save the uploaded image temporarily
    image.save(image_path)

    # Perform face recognition
    face_names = recognize_face(image_path)

    # Load and preprocess the image for sentiment analysis
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, (1, 48, 48, 1)) / 255.0

    # Perform sentiment analysis
    sentiment_labels = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'neutral', 4: 'fear', 5:'surpirse', 6:'disgust'}

    sentiment_prediction = model.predict(img)
    sentiment_label = sentiment_labels[np.argmax(sentiment_prediction)]

    # Check if a face was recognized
    if face_names:
        person_name = face_names[0]  # Assuming only one face is recognized
        result = f"Detected face: {person_name}, Sentiment: {sentiment_label}"
    else:
        result = f"No face detected, Sentiment: {sentiment_label}"

    return result



def recognize_face(image_path):
    # Load the uploaded image

    # Create a new connection for this thread
    conn = sqlite3.connect('database/workers.db')
    cursor = conn.cursor()

    unknown_image = face_recognition.load_image_file(image_path)

    # Find all face locations and encodings in the uploaded image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Initialize variables for face recognition results
    recognized_faces = []

    # Fetch known faces and IDs from the database
    cursor.execute("SELECT individual_id, encoding_data FROM face_encodings")
    known_faces = cursor.fetchall()

    # Extract known face encodings and IDs
    known_face_encodings = [np.frombuffer(face_data, dtype=np.float64) for individual_id, face_data in known_faces]
    known_individual_ids = [individual_id for individual_id, _ in known_faces]

    # Fetch individuals' names from the database
    cursor.execute("SELECT id, name FROM individuals")
    individuals_data = cursor.fetchall()
    individuals_dict = {individual_id: name for individual_id, name in individuals_data}

    # Compare each face in the uploaded image to known faces
    for face_encoding in face_encodings:
        # Compare the face encoding to the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        individual_id = None
        name = "Unknown"  # Default name if no match found

        # If a match is found, use the individual_id from known_individual_ids
        if True in matches:
            first_match_index = matches.index(True)
            individual_id = known_individual_ids[first_match_index]

            # Retrieve the name from the individuals_dict based on individual_id
            name = individuals_dict.get(individual_id, "Unknown")

        recognized_faces.append({"id": individual_id, "name": name})

    conn.close()

    return recognized_faces





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)