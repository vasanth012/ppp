import pickle
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model_path = "model.pickle"
scaler_path = "scaler.pickle"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("✅ Model and scaler loaded successfully!")
else:
    print("❌ Error: Model or scaler file not found!")
    model, scaler = None, None  # Prevent app from crashing if files are missing

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return "Error: Model not loaded properly!"

    # Example: Process user input here
    # Convert input to required format, scale it, and make prediction

    return render_template('result.html', emotion="Happy")  # Dummy response

if __name__ == "__main__":
    app.run(debug=True)
