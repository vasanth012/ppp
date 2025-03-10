import kagglehub
import zipfile
import os
import librosa
import numpy as np

# Step 1: Download & Extract Dataset (Only if not already downloaded)
extract_path = "ravdess_audio"
if not os.path.exists(extract_path):
    print("Downloading RAVDESS dataset...")
    dataset_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

print(f"Dataset available in: {extract_path}")

# Step 2: Load an example audio file and extract features
sample_audio = os.path.join(extract_path, "Actor_01", "03-01-06-01-02-01-01.wav")

if os.path.exists(sample_audio):
    y, sr = librosa.load(sample_audio, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(f"Extracted MFCC shape: {mfccs.shape}")
else:
    print("Sample audio file not found!")
