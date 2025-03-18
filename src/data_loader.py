import pandas as pd
import os
import yfinance as yf

def fetch_and_save_data(stock_symbol):
    try:
        print(f"🌐 Fetching data for {stock_symbol} from Yahoo Finance...")
        stock_data = yf.download(stock_symbol, period="1y", interval="1d")

        if stock_data.empty:
            print(f"❗ Error: Unable to fetch data for {stock_symbol}. Check the symbol or try again.")
            return None

        stock_data.reset_index(inplace=True)
        stock_data.to_csv(f"data/{stock_symbol}.csv", index=False)
        print(f"✅ Data for {stock_symbol} saved to data/{stock_symbol}.csv")
    except Exception as e:
        print(f"❗ Error while fetching data: {e}")

def load_data(stock_symbol):
    file_path = f"data/{stock_symbol}.csv"
    
    if not os.path.exists(file_path):
        print(f"❗ Data for {stock_symbol} not found. Attempting to fetch data...")
        fetch_and_save_data(stock_symbol)
        
        if not os.path.exists(file_path):
            print(f"❗ Failed to fetch or save data for {stock_symbol}.")
            return None
    
    print(f"📥 Loading data for {stock_symbol}...")

    try:
        df = pd.read_csv(file_path)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        required_columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❗ Missing columns in data: {missing_cols}")

        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        print(f"✅ Data for {stock_symbol} loaded successfully! Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"❗ An error occurred while loading data for {stock_symbol}: {e}")
        return None
