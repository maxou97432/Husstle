"""
Orchestrator: fetch → signals → backtest → robustness → verdict GO/NO-GO.

Usage:
    python run_backtest.py [--fetch] [--testnet]
"""
from __future__ import annotations

import argparse
import sys
import os
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

from config import HEDGE_WINDOW, Z_WINDOW as SPREAD_WINDOW, ADF_WINDOW, ADF_STEP, INTERVAL_MS, LOOKBACK_DAYS_DEFAULT

COINS = ["BTC", "ETH"]


def _align_and_aggregate(
    candles_btc: pl.LazyFrame,
    candles_eth: pl.LazyFrame,
    funding_btc: pl.LazyFrame,
    funding_eth: pl.LazyFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Inner-join candles on common timestamps.
    Aggregate funding to 4h windows aligned on candle timestamps.
    Returns (aligned_candles, aligned_funding) as collected DataFrames.
    """
    btc = candles_btc.rename({"close": "close_btc", "open": "open_btc"}).select(["ts", "close_btc"])
    eth = candles_eth.rename({"close": "close_eth", "open": "open_eth"}).select(["ts", "close_eth"])

    candles = btc.join(eth, on="ts", how="inner").sort("ts").collect()

    ts_arr = candles["ts"].to_numpy()

    def agg_funding_to_4h(fund_lf: pl.LazyFrame) -> np.ndarray:
        fund = fund_lf.sort("ts").collect()
        f_ts = fund["ts"].to_numpy()
        f_rate = fund["rate"].to_numpy()
        agg = np.zeros(len(ts_arr), dtype=np.float64)
        interval_ms = INTERVAL_MS
        for i, ts in enumerate(ts_arr):
            mask = (f_ts > ts - interval_ms) & (f_ts <= ts)
            agg[i] = f_rate[mask].sum() if mask.any() else 0.0
        return agg

    fund_btc_arr = agg_funding_to_4h(funding_btc)
    fund_eth_arr = agg_funding_to_4h(funding_eth)

    fund_df = pl.DataFrame({
        "ts": ts_arr,
        "fund_btc": fund_btc_arr,
        "fund_eth": fund_eth_arr,
    })

    return candles, fund_df


def run(fetch: bool, testnet: bool) -> None:
    from data.store import get_conn, load_candles_pl, load_funding_pl
    from data.fetch import fetch_all
    from signal.spread import compute_spread, rolling_zscore
    from signal.cointegration import rolling_adf_pvalue
    from backtest.engine import BacktestConfig, run_backtest
    from backtest.metrics import evaluate
    from backtest.robustness import (
        stress_fees, out_of_sample, bootstrap_thresholds,
        shuffle_test, random_entry_bench,
    )

    if fetch:
        print("=== Fetching data from Hyperliquid ===")
        fetch_all(COINS, lookback_days=LOOKBACK_DAYS_DEFAULT, testnet=testnet)

    conn = get_conn()
    candles_btc = load_candles_pl(conn, "BTC")
    candles_eth = load_candles_pl(conn, "ETH")
    funding_btc = load_funding_pl(conn, "BTC")
    funding_eth = load_funding_pl(conn, "ETH")
    conn.close()

    print("Aligning & aggregating funding…")
    candles, fund_df = _align_and_aggregate(candles_btc, candles_eth, funding_btc, funding_eth)

    n = len(candles)
    if n < HEDGE_WINDOW + SPREAD_WINDOW + 10:
        print(f"ERROR: only {n} common bars (need > {HEDGE_WINDOW + SPREAD_WINDOW + 10}). Run --fetch first.")
        sys.exit(1)

    print(f"=== {n} aligned 4h bars ===")

    close_btc = candles["close_btc"].to_numpy()
    close_eth = candles["close_eth"].to_numpy()
    log_btc = np.log(close_btc)
    log_eth = np.log(close_eth)
    fund_btc_arr = fund_df["fund_btc"].to_numpy()
    fund_eth_arr = fund_df["fund_eth"].to_numpy()

    print("Computing spread (OLS β + α, causal)…")
    spread, beta, alpha = compute_spread(log_eth, log_btc, hedge_window=HEDGE_WINDOW)

    print("Computing z-score…")
    zscore = rolling_zscore(spread, window=SPREAD_WINDOW)

    print("Computing rolling ADF…")
    adf_pval = rolling_adf_pvalue(spread, window=ADF_WINDOW, step=ADF_STEP)

    cfg = BacktestConfig()
    print("Running backtest…")
    trades, equity = run_backtest(
        close_eth, close_btc, spread, zscore, adf_pval,
        fund_eth_arr, fund_btc_arr, beta, cfg,
    )

    print(f"\n=== In-sample results ({len(trades)} trades) ===")
    res = evaluate(trades, equity)
    for k, v in res.items():
        print(f"  {k}: {v}")

    print("\n=== Robustness checks ===")
    args_bt = (close_eth, close_btc, spread, zscore, adf_pval, fund_eth_arr, fund_btc_arr, beta)

    sf = stress_fees(cfg, *args_bt)
    print(f"  stress_fees_2x     PASS={sf['pass']}  sharpe={sf.get('sharpe','n/a')}")
    if not sf['pass']:
        print(f"    → {sf.get('failures')}")

    oos = out_of_sample(cfg, *args_bt)
    print(f"  oos_30pct          PASS={oos['pass']}  sharpe={oos.get('sharpe','n/a')}")
    if not oos['pass']:
        print(f"    → {oos.get('failures')}")

    bt_thresh = bootstrap_thresholds(cfg, *args_bt, n_boot=16, delta=0.20)
    pass_thresh = sum(1 for r in bt_thresh if r["pass"])
    print(f"  thresh_sweep       {pass_thresh}/{len(bt_thresh)} variants pass")

    shuf = shuffle_test(cfg, *args_bt)
    print(f"  shuffle_test       null_sharpe_p95={shuf['null_sharpe_p95']}")

    rand = random_entry_bench(cfg, *args_bt)
    print(f"  random_entry_bench bench_sharpe_p95={rand['bench_sharpe_p95']}")

    real_sharpe = res.get("sharpe", 0.0)
    boot_lo = res.get("bootstrap_sharpe_ci", (0.0, 0.0))[0]

    robust = (
        res["pass"]
        and sf["pass"]
        and oos["pass"]
        and pass_thresh >= len(bt_thresh) * 0.75
        and real_sharpe > shuf["null_sharpe_p95"]
        and real_sharpe > rand["bench_sharpe_p95"]
        and boot_lo > 0
    )

    verdict = "GO" if robust else "NO-GO"
    print(f"\n{'='*42}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*42}")

    if not robust:
        reasons = []
        if not res["pass"]:
            reasons += res.get("failures", [])
        if not sf["pass"]:
            reasons.append("fee stress failed")
        if not oos["pass"]:
            reasons.append("OOS failed")
        if pass_thresh < len(bt_thresh) * 0.75:
            reasons.append(f"threshold sweep: {pass_thresh}/{len(bt_thresh)} pass")
        if real_sharpe <= shuf["null_sharpe_p95"]:
            reasons.append(f"sharpe={real_sharpe} ≤ shuffle null p95={shuf['null_sharpe_p95']}")
        if real_sharpe <= rand["bench_sharpe_p95"]:
            reasons.append(f"sharpe={real_sharpe} ≤ random entry p95={rand['bench_sharpe_p95']}")
        if boot_lo <= 0:
            reasons.append(f"bootstrap CI lower bound={boot_lo} ≤ 0")
        for r in reasons:
            print(f"  ✗ {r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StatArb BTC/ETH backtest orchestrator")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data from Hyperliquid")
    parser.add_argument("--testnet", action="store_true", help="Use Hyperliquid testnet endpoint")
    args = parser.parse_args()
    run(fetch=args.fetch, testnet=args.testnet)
