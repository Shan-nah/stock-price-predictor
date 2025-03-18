import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_data
from predict_visualize import visualize_predictions

def train_model(stock_symbol):
    df = load_data(stock_symbol)
    if df is None:
        return

    # Ensure necessary columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in the data")

    # Features and Target
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Training completed.")

    # Save Model
    joblib.dump(model, 'models/model.pkl')
    print(f"Model saved to models/model.pkl")

    # Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation Results:\nMean Squared Error: {mse}\nR2 Score: {r2}")

    # Visualize Predictions
    visualize_predictions(y_test, y_pred, stock_symbol)

    print("Process completed. Predictions visualized!")

if __name__ == "__main__":
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, TSLA): ").upper()
    train_model(stock_symbol)
