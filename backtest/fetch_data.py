import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def fetch_klines(symbol, interval, start_time, end_time, limit=1500):
    """
    Fetch historical klines from Binance Futures API.
    Handles pagination.
    """
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_klines = []
    
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": limit
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_klines.extend(data)
            
            # Update start_time for the next iteration (timestamp of the last kline + 1ms)
            current_start = datetime.fromtimestamp(data[-1][0] / 1000.0) + timedelta(milliseconds=1)
            
            print(f"Fetched {len(data)} {interval} klines. Total: {len(all_klines)}. Last date: {current_start}")
            
            # Be nice to Binance API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(2) # Wait a bit longer on error
            continue

    return all_klines

def save_to_csv(klines, symbol, interval):
    """
    Convert klines to DataFrame and save to CSV.
    """
    if not klines:
        print(f"No {interval} data to save for {symbol}.")
        return None
        
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Clean up and format
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")
    return filename

def download_historical_data(symbol="ETHUSDT", days=30):
    """
    Downloads both 15m and 1m data for the specified last N days.
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    print(f"--- Fetching Data for {symbol} from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ---")
    
    # Fetch 15m (for signals)
    print("\nFetching 15m klines (Signals)...")
    klines_15m = fetch_klines(symbol, "15m", start_time, end_time)
    save_to_csv(klines_15m, symbol, "15m")
    
    # Fetch 1m (for precise trade simulation)
    print("\nFetching 1m klines (Resolution)...")
    klines_1m = fetch_klines(symbol, "1m", start_time, end_time)
    save_to_csv(klines_1m, symbol, "1m")
    
    print("\n✅ Data collection complete.")

if __name__ == "__main__":
    download_historical_data()
