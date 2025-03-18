import joblib
from sklearn.linear_model import LinearRegression
import os

def create_model():
    """
    Creates a simple linear regression model.
    """
    return LinearRegression()

def save_model(model, filename="models/model.pkl"):
    """
    Saves the trained model using joblib.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename="models/model.pkl"):
    """
    Loads a model from a .pkl file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file '{filename}' not found.")
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
