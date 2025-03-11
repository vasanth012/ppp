from flask import Flask, request, render_template
import os
import librosa
import numpy as np
import pickle  # For loading pre-trained models

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained emotion recognition model
MODEL_PATH = "model.pkl"  # Update with your actual model path
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None  # Ensure the app doesn't break if the model is missing

# Emotion labels (Modify if your model has different labels)
EMOTIONS = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "❌ No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "❌ No selected file!"

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Extract features and predict emotion
    try:
        features = extract_features(file_path)
        if model:
            prediction = model.predict([features])[0]
            predicted_emotion = EMOTIONS[prediction]
        else:
            predicted_emotion = "Unknown (Model not loaded)"
    except Exception as e:
        return f"⚠️ Error processing file: {str(e)}"

    return f"✅ File uploaded successfully: {file.filename}. Predicted Emotion: {predicted_emotion}"

def extract_features(file_path):
    """
    Extract audio features using librosa for emotion classification.
    Modify based on the feature set used in your model.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)
    return mean_mfccs  # Ensure this matches the model’s expected input format

if __name__ == "__main__":
    app.run(debug=True)
