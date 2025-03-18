import pandas as pd
import os

def load_data(stock_symbol):
    file_path = f"data/{stock_symbol}.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❗ Error: Data for {stock_symbol} not found in the data folder.")
        return None
    
    print(f"📥 Fetching data for {stock_symbol}...")

    try:
        # Read CSV, skipping unwanted rows
        df = pd.read_csv(file_path, skiprows=2)

        # Ensure proper column names
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Remove invalid or missing dates
        df = df.dropna(subset=['Date'])

        # Convert numerical columns to float
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with any remaining missing values
        df = df.dropna()

        print(f"✅ Data for {stock_symbol} loaded successfully! Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"❗ An error occurred while loading data for {stock_symbol}: {e}")
        return None
