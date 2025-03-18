import pandas as pd
import joblib
import numpy as np
import os
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
        raise ValueError("❗ Missing required columns in the data")

    # Features and Target
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    print("🛠 Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Training completed.")

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save Model with Stock Symbol
    model_path = f'models/model_{stock_symbol}.pkl'
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

    # Evaluate Model
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📈 Model Evaluation Results:\n🔎 Mean Squared Error: {mse}\n📊 R2 Score: {r2}")

    # Visualize Predictions
    print("🖼 Visualizing predictions...")
    visualize_predictions(y_test, y_pred, stock_symbol)

    print("🎉 Process completed. Predictions visualized!")

if __name__ == "__main__":
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, TSLA): ").upper()
    train_model(stock_symbol)
