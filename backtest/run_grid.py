import os
import sys
import pandas as pd
import time
from itertools import product
from engine import BacktestEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def run_grid_search(symbol="ETHUSDT"):
    print(f"--- Starting MLOps Grid Search for {symbol} ---")
    
    # 1. Define hyperparameter grid
    # To keep it fast, we do a limited grid.
    atr_multipliers = [1.0, 1.2, 1.5, 2.0, 2.5]
    roi_targets_pct = [5.0, 10.0, 15.0, 25.0, 40.0]
    compression_pctiles = [20, 30, 40]
    
    # Generate all combinations
    grid = list(product(atr_multipliers, roi_targets_pct, compression_pctiles))
    print(f"Total parameter combinations to backtest: {len(grid)}")
    
    # 2. To avoid reloading 100 times, load data ONCE
    # The backtest engine is designed to load data once
    base_engine = BacktestEngine(symbol=symbol)
    base_engine.load_data()
    
    all_trades = []
    
    # 3. Run the loop
    start_ts = time.time()
    for idx, (atr_m, roi_t, comp_p) in enumerate(grid):
        print(f"[{idx+1}/{len(grid)}] Testing ATRx{atr_m}, ROI {roi_t}%, Comp {comp_p}%...")
        
        # Configure engine instance for this combination
        engine = BacktestEngine(
            symbol=symbol,
            atr_multiplier=atr_m,
            target_roi_pct=roi_t,
            compression_percentile=comp_p
        )
        
        # Inject the pre-loaded data to save RAM / Time
        engine.df_15m = base_engine.df_15m
        engine.df_1m = base_engine.df_1m
        
        # Run
        try:
            results_df = engine.run()
            if not results_df.empty:
                all_trades.append(results_df)
        except Exception as e:
            print(f"Error skipping this parameter combo: {e}")
            
    # 4. Save to master database (CSV)
    if all_trades:
        master_df = pd.concat(all_trades, ignore_index=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        out_file = os.path.join(DATA_DIR, f"{symbol}_trades_dataset.csv")
        master_df.to_csv(out_file, index=False)
        print(f"\n✅ Grid search complete in {time.time() - start_ts:.1f}s.")
        print(f"Master Dataset created: {out_file} ({len(master_df)} rows)")
        
        # Print a quick summary of the best parameters
        print("\n--- Top 5 Parameter Configurations by Net PnL ---")
        summary = master_df.groupby(['param_sl_mult', 'param_roi_target', 'param_compression_pctile'])['pnl_net'].sum().reset_index()
        summary = summary.sort_values('pnl_net', ascending=False)
        print(summary.head(5).to_string(index=False))
        
    else:
        print("No trades were generated across all parameter combinations.")

if __name__ == "__main__":
    run_grid_search()
