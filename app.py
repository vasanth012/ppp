from flask import Flask, request, render_template
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    # Save file to uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Dummy response (Replace with emotion detection logic)
    return f"✅ File uploaded successfully: {file.filename}. Predicted Emotion: Happy"

if __name__ == "__main__":
    app.run(debug=True)
