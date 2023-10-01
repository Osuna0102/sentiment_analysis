# Sentiment Analysis Project

This project combines sentiment analysis and face recognition to analyze the emotions of individuals in images.

## Technologies Used

- Python
- Flask
- SQLite
- Keras (for sentiment analysis)
- face_recognition (for face recognition)
- OpenCV

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher installed.
- `virtualenv` installed (optional, but recommended for isolated environments).
- Internet connection to download necessary libraries.

## Getting Started

To get the project up and running, follow these steps:

### Installation

1. **Clone the repository:**

```
   git clone https://github.com/Osuna0102/sentiment_analysis.git
   ```

2. **Navigate to the project directory:**
```
   cd your-project
   ```

3. **Create a virtual environment (optional but recommended):**
```
   python -m venv venv
   ```

4. **Activate the virtual environment (Windows):**
   
```
venv\Scripts\activate
   ```

   4.1 Or on macOS and Linux:
```
source venv/bin/activate
   ```

5. **Install the required Python packages:**
```
pip install -r requirements.txt
   ```

Prepare the SQLite database:

6. **Create a new SQLite database or use the provided workers.db if available.**

```
sqlite3 workers.db
   ```
7. **Create the necessary tables for individuals, images, and face_encodings. You can use the following SQL commands:**

```
CREATE TABLE IF NOT EXISTS individuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
    -- Add other columns for additional information about individuals
);
   ```

```
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    individual_id INTEGER NOT NULL,
    image_data BLOB NOT NULL,
    FOREIGN KEY (individual_id) REFERENCES individuals(id)
);
   ```

```
CREATE TABLE IF NOT EXISTS face_encodings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    individual_id INTEGER NOT NULL,
    encoding_data BLOB NOT NULL,
    FOREIGN KEY (individual_id) REFERENCES individuals(id)
);
   ```


8. **Collecting Data**
To collect face data for individuals, use collect_data.py:

```
python collect_data.py
   ```

Follow the prompts to collect images and associate them with individuals.

9. **Training Sentiment Model**
   
To train the sentiment analysis model, use train_sentiment_model.py:

```
python train_sentiment_model.py
   ```

This script will train the model using your dataset. You can customize it further for better performance.


Populate the individuals and images tables with data.

10. **Run the Flask application:**

```
python web_interface.py
   ```

11. **Access the web interface by opening a web browser and navigating to:**


```
http://localhost:8000/
   ```
**Use the web interface to upload an image. The application will perform face recognition and sentiment analysis, displaying the detected person's name and sentiment.

Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

License
This project is licensed under the MIT License - see the LICENSE file for details.
