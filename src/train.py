from src.data_loader import fetch_stock_data
from src.model import build_and_train_model

if __name__ == "__main__":
    ticker = input("Enter Stock Ticker (e.g., AAPL, TSLA): ")
    fetch_stock_data(ticker, start_date="2023-01-01", end_date="2024-01-01")
    build_and_train_model(f"data/{ticker}.csv")
