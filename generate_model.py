import numpy as np
import librosa
import os
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to extract MFCC features
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)  # Shape: (13,)

# Define dataset directory
dataset_dir = "dataset/"
emotions = ["happy", "sad"]  # Modify based on your dataset

# Ensure dataset folder exists
if not os.path.exists(dataset_dir):
    print(f"❌ Error: Dataset folder '{dataset_dir}' not found! Creating it now...")
    os.makedirs(dataset_dir)
    for emotion in emotions:
        os.makedirs(os.path.join(dataset_dir, emotion), exist_ok=True)
    print("✅ Dataset folder created. Please add audio files and rerun the script.")
    exit()  # Stop execution if dataset was missing

data = []
labels = []

for emotion in emotions:
    emotion_path = os.path.join(dataset_dir, emotion)
    
    if not os.path.exists(emotion_path):
        print(f"❌ Error: Folder '{emotion_path}' not found! Skipping...")
        continue

    for file in os.listdir(emotion_path):
        file_path = os.path.join(emotion_path, file)
        features = extract_features(file_path, n_mfcc=13)  # 13 MFCCs
        data.append(features)
        labels.append(emotion)

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

if len(X) == 0:
    print("❌ Error: No data found! Make sure you have audio files in the dataset folder.")
    exit()

print(f"✅ Feature shape: {X.shape}")  # Should be (num_samples, 13)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save model and scaler
with open("model.pickle", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pickle", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and Scaler saved successfully!")
