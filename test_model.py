import os
import numpy as np
import pickle

# File paths
x_test_path = "X_test.npy"
y_test_path = "y_test.npy"
model_path = "model.pickle"

# Check if the files exist
if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
    raise FileNotFoundError("❌ X_test.npy or y_test.npy not found!")

# Load test data
X_test = np.load(x_test_path)
y_test = np.load(y_test_path)

# Load the model
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ model.pickle not found!")
    
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Check feature mismatch
expected_features = model.n_features_in_
actual_features = X_test.shape[1]

if actual_features != expected_features:
    print(f"⚠️ Feature mismatch! Model expects {expected_features}, but X_test has {actual_features}")
    X_test = X_test[:, :expected_features]  # Trim excess features

# Run Prediction
y_pred = model.predict(X_test)
print("✅ Prediction successful!")
