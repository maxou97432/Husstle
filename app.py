#!/usr/bin/env python3
"""
app.py — Flask backend for the Straddle Optimizer Web Dashboard
Wraps all logic from opti.py and exposes a REST API.
"""

import sys
import os
import pickle

# ── Make sure opti.py is importable ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Import all computation functions from opti.py
import opti

# ── ML model paths ────────────────────────────────────────────────────────
_DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
_MODEL_PATH  = os.path.join(_DATA_DIR, 'best_model.pkl')
_SCALER_PATH = os.path.join(_DATA_DIR, 'scaler.pkl')

def _load_ml_model():
    if not os.path.exists(_MODEL_PATH) or not os.path.exists(_SCALER_PATH):
        return None, None
    with open(_MODEL_PATH, 'rb') as f:  model  = pickle.load(f)
    with open(_SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    return model, scaler

_ML_MODEL, _ML_SCALER = _load_ml_model()

app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)


def build_analysis(price_label: str,
                   price_long: float = None,
                   price_short: float = None) -> dict:
    """
    Run full analysis.
    - price_long  : entry price for the LONG leg (Variational)
    - price_short : entry price for the SHORT leg (Paradex)
    If both are given → each leg uses its own price.
    If only one is given or neither → both legs use the same price.
    """
    candles = opti.get_klines(opti.SYMBOL, opti.KLINE_INTERVAL, opti.KLINE_LIMIT)
    comp    = opti.detect_compression(candles)

    # Strategy %s derived from midpoint (or single price)
    mid_price = (price_long + price_short) / 2 if (price_long and price_short) else (price_long or price_short)
    trade     = opti.calculate_trade(mid_price, comp["atr_now"])

    sl_pct = trade["sl_pct"]
    tp_pct = trade["tp_pct"]

    pl = price_long  or mid_price
    ps = price_short or mid_price

    # Per-leg TP/SL applied to each leg's own entry price
    long_tp  = round(pl * (1 + tp_pct), 4)
    long_sl  = round(pl * (1 - sl_pct), 4)
    short_tp = round(ps * (1 - tp_pct), 4)
    short_sl = round(ps * (1 + sl_pct), 4)

    # Spread between the two DEX prices
    spread     = round(abs(pl - ps), 4)
    spread_pct = round(spread / min(pl, ps) * 100, 4) if min(pl, ps) > 0 else 0.0

    return {
        "meta": {
            "symbol":        opti.SYMBOL,
            "leverage":      opti.LEVERAGE,
            "interval":      opti.KLINE_INTERVAL,
            "kline_limit":   opti.KLINE_LIMIT,
            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price_label":   price_label,
            "params": {
                "symbol":                  opti.SYMBOL,
                "position_size_usdc":      opti.POSITION_SIZE_USDC,
                "leverage":                opti.LEVERAGE,
                "target_roi_collateral":   opti.TARGET_ROI_COLLATERAL,
                "max_loss_pct_collateral": opti.MAX_LOSS_PCT_COLLATERAL,
                "atr_multiplier":          opti.ATR_MULTIPLIER,
                "atr_period":              opti.ATR_PERIOD,
                "compression_percentile":  opti.COMPRESSION_PERCENTILE,
                "kline_interval":          opti.KLINE_INTERVAL,
                "kline_limit":             opti.KLINE_LIMIT,
                "min_rr_ratio":            opti.MIN_RR_RATIO,
            }
        },
        "compression": {
            "score":          comp["score"],
            "verdict":        comp["verdict"],
            "verdict_color":  comp["verdict_color"],
            "go":             comp["go"],
            "atr_now":        round(comp["atr_now"], 5),
            "atr_thresh":     round(comp["atr_thresh"], 5),
            "atr_rank":       round(comp["atr_rank"], 1),
            "atr_compressed": comp["atr_compressed"],
            "bbw_now":        round(comp["bbw_now"] * 100, 3),
            "bbw_thresh":     round(comp["bbw_thresh"] * 100, 3),
            "bbw_rank":       round(comp["bbw_rank"], 1),
            "bbw_compressed": comp["bbw_compressed"],
            "history_atr":    comp["history_atr"],
            "history_bbw":    [b * 100 for b in comp["history_bbw"]],
        },
        "trade": {
            "position_size_usdc": opti.POSITION_SIZE_USDC,
            "collateral":         round(trade["collateral"], 2),
            "profit_target":      round(trade["profit_target"], 2),
            "entry":              round(mid_price, 4),
            "entry_long":         round(pl, 4),
            "entry_short":        round(ps, 4),
            "tp_pct":             round(tp_pct * 100, 3),
            "sl_pct":             round(sl_pct * 100, 3),
            "win_amount":         round(trade["win_amount"], 2),
            "loss_amount":        round(trade["loss_amount"], 2),
            "net_profit":         round(trade["net_profit"], 2),
            "rr_ratio":           round(trade["rr_ratio"], 2),
            "roi_tp_pct":         round(trade["roi_tp_pct"], 1),
            "roi_sl_pct":         round(trade["roi_sl_pct"], 1),
            "roi_net":            round(trade["roi_net"], 1),
            "long_tp":            long_tp,
            "long_sl":            long_sl,
            "short_tp":           short_tp,
            "short_sl":           short_sl,
            "liq_dist":           round((1 / opti.LEVERAGE) * mid_price, 2),
            "min_rr_ratio":       opti.MIN_RR_RATIO,
            "target_roi_pct":     opti.TARGET_ROI_COLLATERAL * 100,
            "spread":             spread,
            "spread_pct":         spread_pct,
            "split_prices":       (price_long is not None or price_short is not None),
        },
    }


@app.route("/api/analyze", methods=["GET", "POST"])
def analyze():
    """
    POST → {
        "time":        "HH:MM:SS",   # optional — historical Binance price
        "price_long":  2074.50,       # optional — manual LONG entry (Variational)
        "price_short": 2075.20,       # optional — manual SHORT entry (Paradex)
        "params": { ... }             # optional — override any config value
    }
    Priority: price_long/price_short > time > live
    """
    try:
        body = request.get_json(silent=True) or {} if request.method == "POST" else {}

        # ── Apply parameter overrides ──────────────────────────────────────────
        params = body.get("params", {})

        PARAM_MAP = {
            "symbol":                  ("SYMBOL",                 str),
            "position_size_usdc":      ("POSITION_SIZE_USDC",     float),
            "leverage":                ("LEVERAGE",               int),
            "target_roi_collateral":   ("TARGET_ROI_COLLATERAL",  float),
            "max_loss_pct_collateral": ("MAX_LOSS_PCT_COLLATERAL", float),
            "atr_multiplier":          ("ATR_MULTIPLIER",          float),
            "atr_period":              ("ATR_PERIOD",              int),
            "compression_percentile":  ("COMPRESSION_PERCENTILE",  int),
            "kline_interval":          ("KLINE_INTERVAL",          str),
            "kline_limit":             ("KLINE_LIMIT",             int),
            "min_rr_ratio":            ("MIN_RR_RATIO",            float),
        }

        originals = {attr: getattr(opti, attr) for _, (attr, _) in PARAM_MAP.items()}

        try:
            for key, (attr, cast) in PARAM_MAP.items():
                if key in params:
                    setattr(opti, attr, cast(params[key]))

            # ── Determine entry prices ─────────────────────────────────────────
            price_long_raw  = body.get("price_long")
            price_short_raw = body.get("price_short")

            price_long  = float(price_long_raw)  if price_long_raw  else None
            price_short = float(price_short_raw) if price_short_raw else None

            if price_long or price_short:
                # Option B : prix réels saisis manuellement
                if price_long and price_short:
                    label = f"Manuel LONG={price_long} / SHORT={price_short}"
                elif price_long:
                    label = f"Manuel LONG={price_long}"
                else:
                    label = f"Manuel SHORT={price_short}"

            else:
                # Option A : heure Binance → prix historique
                time_str = body.get("time", "").strip()
                if time_str:
                    now = datetime.now()
                    t   = datetime.strptime(time_str, "%H:%M:%S").replace(
                        year=now.year, month=now.month, day=now.day
                    )
                    binance_price = opti.get_price_at_time(opti.SYMBOL, t)
                    price_long = price_short = binance_price
                    label = f"Historique ({t.strftime('%H:%M:%S')})"
                else:
                    binance_price = opti.get_live_price(opti.SYMBOL)
                    price_long = price_short = binance_price
                    label = f"LIVE ({datetime.now().strftime('%H:%M:%S')})"

            data = build_analysis(label, price_long, price_short)

        finally:
            for attr, val in originals.items():
                setattr(opti, attr, val)

        return jsonify({"ok": True, "data": data})

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/ml-suggest", methods=["GET"])
def ml_suggest():
    """
    Returns the best parameter set predicted by the ML model
    given the current live market state (atr_rank, bbw_rank, atr_now).
    """
    try:
        if _ML_MODEL is None:
            return jsonify({"ok": False, "error": "Modèle ML non trouvé. Relancez backtest/train_model.py."}), 404

        import numpy as np

        # Fetch live market state
        candles = opti.get_klines(opti.SYMBOL, opti.KLINE_INTERVAL, opti.KLINE_LIMIT)
        comp    = opti.detect_compression(candles)

        atr_now  = comp["atr_now"]
        atr_rank = comp["atr_rank"]
        bbw_rank = comp["bbw_rank"]

        # Build candidate grid
        param_grid = [
            {"param_sl_mult": sl_m, "param_roi_target": roi, "param_compression_pctile": cp}
            for sl_m in [1.0, 1.2, 1.5, 2.0, 2.5]
            for roi  in [0.05, 0.10, 0.15, 0.25, 0.40]
            for cp   in [20, 30, 40]
        ]

        feature_cols = ['atr_now', 'bbw_now', 'atr_rank',
                        'param_sl_mult', 'param_roi_target', 'param_compression_pctile']

        rows = []
        for p in param_grid:
            rows.append([
                atr_now, bbw_rank, atr_rank,
                p["param_sl_mult"], p["param_roi_target"], p["param_compression_pctile"]
            ])

        X = np.array(rows)
        X_scaled = _ML_SCALER.transform(X)
        preds = _ML_MODEL.predict(X_scaled)

        best_idx = int(np.argmax(preds))
        best = param_grid[best_idx]

        return jsonify({
            "ok": True,
            "suggestion": {
                "atr_multiplier":         best["param_sl_mult"],
                "target_roi_collateral":  best["param_roi_target"],
                "compression_percentile": best["param_compression_pctile"],
                "predicted_pnl":          round(float(preds[best_idx]), 2),
                "market_state": {
                    "atr_now":  round(atr_now, 5),
                    "atr_rank": round(atr_rank, 1),
                    "bbw_rank": round(bbw_rank, 1),
                }
            }
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


import json
import uuid

_TRADE_LOG_PATH = os.path.join(_DATA_DIR, 'trade_log.json')

def _load_trades():
    if not os.path.exists(_TRADE_LOG_PATH):
        return []
    try:
        with open(_TRADE_LOG_PATH, 'r') as f:
            return json.load(f)
    except:
        return []

def _save_trades(trades):
    with open(_TRADE_LOG_PATH, 'w') as f:
        json.dump(trades, f, indent=2)

@app.route("/api/trades", methods=["GET", "POST", "DELETE"])
def api_trades():
    if request.method == "GET":
        trades = _load_trades()
        # sort by date desc
        trades.sort(key=lambda x: x.get('date', ''), reverse=True)
        return jsonify({"ok": True, "data": trades})

    elif request.method == "POST":
        trades = _load_trades()
        body = request.get_json(silent=True) or {}
        new_trade = {
            "id": str(uuid.uuid4()),
            "date": body.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "symbol": body.get("symbol", "ETHUSDT"),
            "entry_long": float(body.get("entry_long", 0)),
            "entry_short": float(body.get("entry_short", 0)),
            "pnl": float(body.get("pnl", 0)),
            "notes": body.get("notes", ""),
            "status": body.get("status", "CLOSED") # OPEN, WON, LOST
        }
        trades.append(new_trade)
        _save_trades(trades)
        return jsonify({"ok": True, "trade": new_trade})
        
    elif request.method == "DELETE":
        trade_id = request.args.get("id")
        if not trade_id:
            return jsonify({"ok": False, "error": "Missing ID"}), 400
            
        trades = _load_trades()
        trades = [t for t in trades if t.get("id") != trade_id]
        _save_trades(trades)
        return jsonify({"ok": True})


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    print("\n  ⚡  Straddle Optimizer — Web Dashboard")
    print(f"  🌐  Dashboard : http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
