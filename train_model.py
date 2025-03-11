import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dummy training data (Replace with actual feature extraction)
X_train = np.random.rand(100, 13)  # 100 samples, 13 features
y_train = np.random.randint(0, 4, 100)  # 4 emotion classes

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model correctly
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved successfully!")
