import face_recognition
import numpy as np
import sqlite3
import cv2  # Import OpenCV

# Connect to the database
conn = sqlite3.connect('database\workers.db')
cursor = conn.cursor()

# Fetch all the image data from the images table
cursor.execute("SELECT individual_id, image_data FROM images")
image_records = cursor.fetchall()

# Initialize a list to store face encodings and corresponding individual IDs
face_encodings = []

# Iterate through the image records
for individual_id, image_data in image_records:
    # Load the image data as a numpy array
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Find face encodings for all faces in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings_batch = face_recognition.face_encodings(image, face_locations)

    # Append the face encodings and individual IDs to the list
    for encoding in face_encodings_batch:
        face_encodings.append((individual_id, encoding.tobytes()))

# Insert the face encodings into the face_encodings table
cursor.executemany("INSERT INTO face_encodings (individual_id, encoding_data) VALUES (?, ?)", face_encodings)

# Commit the changes and close the database connection
conn.commit()
conn.close()
