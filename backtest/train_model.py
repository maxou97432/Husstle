"""
train_model.py — MLOps: Train a model to predict the best parameters
for the Delta-Neutral Straddle strategy based on market state.
"""

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def load_and_prepare_data(csv_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(df.head(3))

    # ── Features: market state at the moment of signal ──
    feature_cols = [
        'atr_now',               # Raw ATR value (volatility)
        'bbw_now',               # BB Width rank (0–100%)
        'atr_rank',              # ATR percentile rank (0–100%)
        'param_sl_mult',         # ATR multiplier for SL
        'param_roi_target',      # ROI target (as decimal)
        'param_compression_pctile', # Compression percentile threshold
    ]

    target_col = 'pnl_net'

    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols].values
    y = df[target_col].values

    return X, y, feature_cols, df

def train(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, 'ETHUSDT_trades_dataset.csv')

    X, y, feature_cols, df = load_and_prepare_data(csv_path)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (helps some models)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Model 1: Gradient Boosting (usually best for tabular data) ──
    print("\nTraining Gradient Boosting Regressor...")
    gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    gbr.fit(X_train_s, y_train)

    y_pred_gbr = gbr.predict(X_test_s)
    r2_gbr  = r2_score(y_test, y_pred_gbr)
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
    print(f"  GBR  → R² = {r2_gbr:.4f}  |  MAE = ${mae_gbr:.4f}")

    # ── Model 2: Random Forest ──
    print("Training Random Forest Regressor...")
    rfr = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rfr.fit(X_train_s, y_train)

    y_pred_rfr = rfr.predict(X_test_s)
    r2_rfr  = r2_score(y_test, y_pred_rfr)
    mae_rfr = mean_absolute_error(y_test, y_pred_rfr)
    print(f"  RFR  → R² = {r2_rfr:.4f}  |  MAE = ${mae_rfr:.4f}")

    # ── Feature Importance ──
    best_model = gbr if r2_gbr >= r2_rfr else rfr
    best_scaler = scaler
    print(f"\n✅ Best model: {'GBR' if r2_gbr >= r2_rfr else 'RFR'}")

    importances = best_model.feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<32} {bar} {imp:.4f}")

    # ── Optimizer: what params maximize expected PnL for a given market state? ──
    print("\nRunning parameter optimizer on dataset...")
    best_params_df = find_best_params(df, best_model, best_scaler, feature_cols)
    print(best_params_df.to_string(index=False))

    # ── Save model + scaler ──
    model_path  = os.path.join(MODEL_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(best_scaler, f)
    print(f"\nModel saved → {model_path}")
    print(f"Scaler saved → {scaler_path}")

    return best_model, best_scaler, feature_cols

def find_best_params(df, model, scaler, feature_cols,
                     atr_mult_candidates=[1.0, 1.2, 1.5, 2.0, 2.5],
                     roi_candidates=[0.05, 0.10, 0.15, 0.25, 0.40],
                     comp_candidates=[20, 30, 40]):
    """
    For a given current market state, predict which parameter set will
    yield the highest expected PnL, using the trained model.
    """
    # Use the mean market state as a representative example
    mean_atr     = df['atr_now'].mean()
    mean_bbw     = df['bbw_now'].mean()
    mean_atr_rank = df['atr_rank'].mean()

    rows = []
    for sl_m in atr_mult_candidates:
        for roi in roi_candidates:
            for comp in comp_candidates:
                rows.append({
                    'atr_now': mean_atr,
                    'bbw_now': mean_bbw,
                    'atr_rank': mean_atr_rank,
                    'param_sl_mult': sl_m,
                    'param_roi_target': roi,
                    'param_compression_pctile': comp,
                })

    cand_df = pd.DataFrame(rows)[feature_cols]
    cand_scaled = scaler.transform(cand_df.values)
    preds = model.predict(cand_scaled)

    cand_df['predicted_pnl'] = preds
    top5 = cand_df.sort_values('predicted_pnl', ascending=False).head(5)
    return top5

def predict_best_params(atr_now, atr_rank, bbw_rank,
                        model_path=None, scaler_path=None):
    """
    Public API: Given current live market state, returns the best
    parameter set predicted by the ML model.
    Called by app.py for live suggestions.
    """
    if model_path is None:
        model_path  = os.path.join(MODEL_DIR, 'best_model.pkl')
    if scaler_path is None:
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    param_grid = [
        {'param_sl_mult': sl_m, 'param_roi_target': roi, 'param_compression_pctile': comp}
        for sl_m in [1.0, 1.2, 1.5, 2.0, 2.5]
        for roi in [0.05, 0.10, 0.15, 0.25, 0.40]
        for comp in [20, 30, 40]
    ]

    rows = []
    for p in param_grid:
        rows.append({
            'atr_now': atr_now,
            'bbw_now': bbw_rank,
            'atr_rank': atr_rank,
            **p
        })

    feature_cols = ['atr_now', 'bbw_now', 'atr_rank',
                    'param_sl_mult', 'param_roi_target', 'param_compression_pctile']
    df_cand = pd.DataFrame(rows)[feature_cols]
    X_scaled = scaler.transform(df_cand.values)
    preds = model.predict(X_scaled)

    best_idx = int(np.argmax(preds))
    best = param_grid[best_idx]
    best['predicted_pnl'] = float(preds[best_idx])
    return best

if __name__ == "__main__":
    train()
