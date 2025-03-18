import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def visualize_predictions(y_test, y_pred, stock_symbol, X_test):
    # Plot Actual vs Predicted with Dates
    dates = pd.date_range(end=pd.Timestamp.today(), periods=len(y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test, label="Actual Prices", color="blue")
    plt.plot(dates, y_pred, label="Predicted Prices", color="red", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title(f"Actual vs Predicted Prices for {stock_symbol}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/{stock_symbol}_prediction.png')
    plt.show()
    print(f"📊 Visualization saved as visualizations/{stock_symbol}_prediction.png")

def plot_moving_averages(df, stock_symbol):
    # Calculate Moving Averages
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()

    # Plot Moving Averages
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['MA30'], label='30-Day MA', color='green')
    plt.plot(df['Date'], df['MA100'], label='100-Day MA', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{stock_symbol} Stock Prices with Moving Averages')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'visualizations/{stock_symbol}_moving_averages.png')
    plt.show()
    print(f"📈 Moving Averages plot saved as visualizations/{stock_symbol}_moving_averages.png")

def plot_correlation_heatmap(df, stock_symbol):
    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[['Close', 'High', 'Low', 'Open', 'Volume']].corr(), annot=True, cmap='coolwarm')
    plt.title(f'{stock_symbol} Correlation Heatmap')
    plt.savefig(f'visualizations/{stock_symbol}_correlation_heatmap.png')
    plt.show()
    print(f"🔥 Correlation Heatmap saved as visualizations/{stock_symbol}_correlation_heatmap.png")
