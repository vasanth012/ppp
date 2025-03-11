import numpy as np
import pickle

# Load and fix X_test
X_test = np.load("X_test.npy")
X_test_fixed = X_test[:, :13]  # Keep only the first 13 features
np.save("X_test_fixed.npy", X_test_fixed)
print("✅ Fixed X_test.npy saved as X_test_fixed.npy")

# Load model
with open("model.pickle", "rb") as f:                                          
    model = pickle.load(f)

# Make predictions
y_pred = model.predict(X_test_fixed)
print("Predictions:", y_pred)
