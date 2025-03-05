import librosa
import numpy as np
import os
import pickle

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

dataset_folder = "data/"
features, labels = [], []

for file in os.listdir(dataset_folder):
    if file.endswith(".wav"):
        emotion = file.split("_")[2]  # Assuming filename pattern: actor_speaker_emotion.wav
        features.append(extract_features(os.path.join(dataset_folder, file)))
        labels.append(emotion)

features = np.array(features)
labels = np.array(labels)

# Save extracted features and labels
pickle.dump((features, labels), open("features.pickle", "wb"))
