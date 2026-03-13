import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Removed opti.py imports since we use vectorized pandas exactly matching its logic


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

class BacktestEngine:
    def __init__(self, symbol="ETHUSDT", 
                 atr_period=14, atr_multiplier=1.5,
                 compression_percentile=30,
                 target_roi_pct=15.0, leverage=50,
                 position_size=4000,
                 fee_rate=0.0004): # Assume 0.04% maker/taker avg fee
        self.symbol = symbol
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.compression_percentile = compression_percentile
        
        # Trade parameters
        self.target_roi_pct = target_roi_pct / 100.0  # e.g. 0.15 for 15% ROI
        self.leverage = leverage
        self.position_size = position_size
        self.fee_rate = fee_rate
        
        self.df_15m = None
        self.df_1m = None
        self.trades = []

    def load_data(self):
        """Loads and pre-processes CSV data."""
        f_15m = os.path.join(DATA_DIR, f"{self.symbol}_15m.csv")
        f_1m = os.path.join(DATA_DIR, f"{self.symbol}_1m.csv")
        
        if not os.path.exists(f_15m) or not os.path.exists(f_1m):
            raise FileNotFoundError(f"Missing data files. Run fetch_data.py first.")
            
        print(f"Loading data from {DATA_DIR}...")
        self.df_15m = pd.read_csv(f_15m, parse_dates=['timestamp'])
        self.df_1m = pd.read_csv(f_1m, parse_dates=['timestamp'])
        
        # Ensure sorting
        self.df_15m.sort_values('timestamp', inplace=True)
        self.df_1m.sort_values('timestamp', inplace=True)
        self.df_1m.set_index('timestamp', inplace=True)
        
        print(f"Loaded {len(self.df_15m)} 15m candles, and {len(self.df_1m)} 1m candles.")
        self._calculate_indicators()

    def _calculate_indicators(self):
        """Computes ATR and BB Width on the 15m dataframe."""
        # Convert to list of dicts format expected by opti.py helpers
        # (Though we can do it faster in pandas, we reuse the exact logic to be perfectly faithful)
        # But for backtesting 30 days (2880 rows), pandas vectorized is much better.
        # Let's write the vectorized versions that exactly match opti.py
        
        df = self.df_15m
        
        # ATR Calculation
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['prev_close']).abs()
        df['tr3'] = (df['low'] - df['prev_close']).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Opti.py uses simple moving average for ATR
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        # Bollinger Bands Width (20, 2)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std(ddof=0) # opti.py uses population std dev roughly
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20'] * 100
        
        # Clean up NaNs from rolling windows
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

    def run(self):
        """Runs the backtest loop."""
        print(f"Starting simulation for {self.symbol}...")
        print(f"Params: ROI {self.target_roi_pct*100}%, ATRx{self.atr_multiplier}, Comp {self.compression_percentile}th pctile")
        
        df = self.df_15m
        trades_recorded = 0
        
        # We start at index 150 because we need a 150-candle history to compute percentiles
        for i in range(150, len(df) - 1):
            current_row = df.iloc[i]
            timestamp = current_row['timestamp']
            
            # The signal is evaluated on the CLOSE of the current 15m candle.
            # Entry will happen exactly at this close price (or the open of the next candle)
            entry_price = current_row['close']
            
            # 1. Analyze Compression (Lookback window of 150 candles ending AT current candle)
            window = df.iloc[i-149 : i+1]
            
            atr_now = current_row['atr']
            bbw_now = current_row['bb_width']
            
            atr_thresh = np.percentile(window['atr'], self.compression_percentile)
            bbw_thresh = np.percentile(window['bb_width'], self.compression_percentile)
            
            is_compressed = (atr_now <= atr_thresh) and (bbw_now <= bbw_thresh)
            
            if is_compressed:
                # WE HAVE A SIGNAL! Fire the trade.
                self._simulate_trade(
                    entry_time=timestamp, 
                    entry_price=entry_price,
                    atr_now=atr_now,
                    atr_rank=self._calc_rank(atr_now, window['atr']),
                    bbw_rank=self._calc_rank(bbw_now, window['bb_width'])
                )
                trades_recorded += 1
                
        print(f"Simulation complete. {trades_recorded} trades recorded.")
        return pd.DataFrame(self.trades)
        
    def _calc_rank(self, val, series):
        """Percentile rank of val within series."""
        count_below = np.sum(series < val)
        return (count_below / len(series)) * 100

    def _simulate_trade(self, entry_time, entry_price, atr_now, atr_rank, bbw_rank):
        """Resolves the outcome of a trade using 1m data."""
        
        # Calculate levels exactly like app.py / opti.py
        # Both Long and Short have the same entry price in this base simulation (no spread)
        margin = self.position_size / self.leverage
        profit_target = margin * self.target_roi_pct
        
        tp_pct = (profit_target / self.position_size)
        sl_dist = atr_now * self.atr_multiplier
        sl_pct = sl_dist / entry_price
        
        # LONG Leg
        long_tp = entry_price * (1 + tp_pct)
        long_sl = entry_price - sl_dist
        
        # SHORT Leg
        short_tp = entry_price * (1 - tp_pct)
        short_sl = entry_price + sl_dist
        
        # Resolution phase: Scan 1m candles starting from the very next minute
        # to see which hits first: TP or SL for *each* leg independently.
        future_1m = self.df_1m.loc[entry_time + pd.Timedelta(minutes=1) :]
        
        long_outcome = {"status": "OPEN", "close_time": None, "pnl": 0}
        short_outcome = {"status": "OPEN", "close_time": None, "pnl": 0}
        
        for idx, row in future_1m.iterrows():
            high = row['high']
            low = row['low']
            
            # Resolve LONG
            if long_outcome["status"] == "OPEN":
                # Did it hit SL first in this minute? (Pessimistic: assume SL hit before TP if both in same candle)
                if low <= long_sl:
                    long_outcome.update({"status": "SL", "close_time": idx, "pnl": -self.position_size * sl_pct})
                elif high >= long_tp:
                    long_outcome.update({"status": "TP", "close_time": idx, "pnl": profit_target})
                    
            # Resolve SHORT
            if short_outcome["status"] == "OPEN":
                # Did it hit SL first? (For short, SL is above)
                if high >= short_sl:
                    short_outcome.update({"status": "SL", "close_time": idx, "pnl": -self.position_size * sl_pct})
                elif low <= short_tp:
                    short_outcome.update({"status": "TP", "close_time": idx, "pnl": profit_target})
                    
            if long_outcome["status"] != "OPEN" and short_outcome["status"] != "OPEN":
                break # Both legs resolved
                
        # Handle trades that never resolved (end of dataset)
        if long_outcome["status"] == "OPEN":
            # Force close at last known price
            last_price = future_1m.iloc[-1]['close'] if len(future_1m) > 0 else entry_price
            p_pct = (last_price - entry_price) / entry_price
            long_outcome.update({"status": "TIMEOUT", "close_time": future_1m.index[-1] if len(future_1m)>0 else entry_time, "pnl": self.position_size * p_pct})

        if short_outcome["status"] == "OPEN":
            last_price = future_1m.iloc[-1]['close'] if len(future_1m) > 0 else entry_price
            p_pct = (entry_price - last_price) / entry_price
            short_outcome.update({"status": "TIMEOUT", "close_time": future_1m.index[-1] if len(future_1m)>0 else entry_time, "pnl": self.position_size * p_pct})
            
        # Fees: Apply entry and exit fee for both legs
        fees = (self.position_size * self.fee_rate * 2) * 2 # 2 legs, 2 events (open/close)
        
        net_pnl = long_outcome["pnl"] + short_outcome["pnl"] - fees
        
        win_rate = 0
        if long_outcome["status"] == "TP" and short_outcome["status"] == "TP": win_rate = 1.0
        elif long_outcome["status"] == "TP" or short_outcome["status"] == "TP": win_rate = 0.5
        
        # Log the trade
        self.trades.append({
            "entry_time": entry_time,
            "symbol": self.symbol,
            "atr_now": atr_now,
            "bbw_now": bbw_rank, # Storing rank is often more useful for ML than raw BBW
            "atr_rank": atr_rank,
            "param_sl_mult": self.atr_multiplier,
            "param_roi_target": self.target_roi_pct,
            "param_compression_pctile": self.compression_percentile,
            "long_status": long_outcome["status"],
            "short_status": short_outcome["status"],
            "pnl_long": long_outcome["pnl"],
            "pnl_short": short_outcome["pnl"],
            "fees": fees,
            "pnl_net": net_pnl,
            "win_rate": win_rate
        })

if __name__ == "__main__":
    engine = BacktestEngine(symbol="ETHUSDT")
    try:
        engine.load_data()
        results_df = engine.run()
        print(f"\nBacktest Results Snapshot:")
        print(results_df[['entry_time', 'long_status', 'short_status', 'pnl_net']].head(10))
        print(f"\nTotal Net PnL over {len(results_df)} trades: ${results_df['pnl_net'].sum():.2f}")
    except Exception as e:
        print(f"Error: {e}")
