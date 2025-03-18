import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.to_csv(f"data/{ticker}.csv")
    print(f"Data saved to data/{ticker}.csv")
