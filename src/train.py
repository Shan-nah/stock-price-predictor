import os
import pandas as pd
from data_loader import load_data
from predict_visualize import visualize_predictions, plot_moving_averages, plot_correlation_heatmap

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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Training completed.")

    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    import joblib
    joblib.dump(model, f'models/{stock_symbol}_model.pkl')
    print(f"Model saved to models/{stock_symbol}_model.pkl")

    # Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation Results:\nMean Squared Error: {mse}\nR2 Score: {r2}")

    # Ensure Visualization Directory
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # Visualize Predictions
    visualize_predictions(y_test, y_pred, stock_symbol, X_test)

    # Plot Moving Averages
    plot_moving_averages(df, stock_symbol)

    # Plot Correlation Heatmap
    plot_correlation_heatmap(df, stock_symbol)

    print("✅ All visualizations completed!")

if __name__ == "__main__":
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, TSLA): ").upper()
    train_model(stock_symbol)
