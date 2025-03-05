import pickle

with open("model.pickle", "rb") as file:
    model = pickle.load(file)

print("✅ Model loaded successfully!")
print("Expected feature size:", model.n_features_in_)
