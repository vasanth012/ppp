import kagglehub
import shutil
import os

# Step 1: Download dataset
dataset_path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

# Step 2: Define the target folder
extract_path = "ravdess_audio"

# Step 3: Move dataset to the correct location
if not os.path.exists(extract_path):
    shutil.copytree(dataset_path, extract_path)
    print(f"✅ Dataset copied to: {extract_path}")
else:
    print(f"⚠️ Dataset already exists at: {extract_path}")
