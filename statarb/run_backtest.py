"""
Orchestrator: fetch data → compute signals → backtest → robustness → verdict GO/NO-GO.

Usage:
    python run_backtest.py [--fetch] [--testnet]
"""
from __future__ import annotations

import argparse
import sys
import numpy as np

COINS = ["BTC", "ETH"]
HEDGE_WINDOW = 240
SPREAD_WINDOW = 60
ADF_WINDOW = 240
ADF_STEP = 6


def _align_funding(candle_ts: np.ndarray, funding_rows: list[dict]) -> np.ndarray:
    """Map hourly funding to 4h candle timestamps via nearest-prior lookup."""
    f_ts = np.array([r["ts"] for r in funding_rows], dtype=np.int64)
    f_rate = np.array([r["rate"] for r in funding_rows], dtype=np.float64)
    agg = np.zeros(len(candle_ts), dtype=np.float64)
    for i, ts in enumerate(candle_ts):
        # sum funding rates for the 4 hourly periods ending at this candle
        mask = (f_ts > ts - 4 * 3600 * 1000) & (f_ts <= ts)
        agg[i] = f_rate[mask].sum() if mask.any() else 0.0
    return agg


def run(fetch: bool, testnet: bool) -> None:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from data.store import get_conn, load_candles, load_funding
    from data.fetch import fetch_all
    from signal.hedge import rolling_ols_hedge
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
        fetch_all(COINS, lookback_days=365, testnet=testnet)

    conn = get_conn()
    btc_rows = load_candles(conn, "BTC")
    eth_rows = load_candles(conn, "ETH")
    btc_fund_rows = load_funding(conn, "BTC")
    eth_fund_rows = load_funding(conn, "ETH")
    conn.close()

    if not btc_rows or not eth_rows:
        print("ERROR: no candle data. Run with --fetch first.")
        sys.exit(1)

    # Align on common timestamps
    btc_ts = {r["ts"]: r for r in btc_rows}
    eth_ts = {r["ts"]: r for r in eth_rows}
    common = sorted(set(btc_ts) & set(eth_ts))

    if len(common) < HEDGE_WINDOW + SPREAD_WINDOW + 10:
        print(f"ERROR: only {len(common)} common bars (need >{HEDGE_WINDOW + SPREAD_WINDOW + 10})")
        sys.exit(1)

    ts_arr = np.array(common, dtype=np.int64)
    close_btc = np.array([btc_ts[t]["close"] for t in common])
    close_eth = np.array([eth_ts[t]["close"] for t in common])

    log_btc = np.log(close_btc)
    log_eth = np.log(close_eth)

    fund_btc = _align_funding(ts_arr, btc_fund_rows)
    fund_eth = _align_funding(ts_arr, eth_fund_rows)

    print(f"=== {len(common)} aligned 4h bars ===")
    print("Computing hedge ratio…")
    beta = rolling_ols_hedge(log_eth, log_btc, window=HEDGE_WINDOW)

    print("Computing spread & z-score…")
    spread = compute_spread(log_eth, log_btc, hedge_window=HEDGE_WINDOW)
    zscore = rolling_zscore(spread, window=SPREAD_WINDOW)

    print("Computing rolling ADF…")
    adf_pval = rolling_adf_pvalue(spread, window=ADF_WINDOW, step=ADF_STEP)

    cfg = BacktestConfig()
    print("Running backtest…")
    trades, equity = run_backtest(
        close_eth, close_btc, spread, zscore, adf_pval,
        fund_eth, fund_btc, beta, cfg,
    )

    print(f"\n=== In-sample results ({len(trades)} trades) ===")
    res = evaluate(trades, equity)
    for k, v in res.items():
        print(f"  {k}: {v}")

    print("\n=== Robustness checks ===")

    args_bt = (close_eth, close_btc, spread, zscore, adf_pval, fund_eth, fund_btc, beta)

    sf = stress_fees(cfg, *args_bt)
    print(f"  stress_fees_2x  pass={sf['pass']}  sharpe={sf.get('sharpe', 'n/a'):.2f}")

    oos = out_of_sample(cfg, *args_bt)
    print(f"  oos_30pct       pass={oos['pass']}  sharpe={oos.get('sharpe', 'n/a'):.2f}")

    bt_thresh = bootstrap_thresholds(cfg, *args_bt, n_boot=16, delta=0.20)
    pass_thresh = sum(1 for r in bt_thresh if r["pass"])
    print(f"  thresh_sweep    {pass_thresh}/{len(bt_thresh)} variants pass")

    shuf = shuffle_test(cfg, *args_bt)
    print(f"  shuffle_test    null_sharpe_p95={shuf['null_sharpe_p95']:.2f}")

    rand = random_entry_bench(cfg, *args_bt)
    print(f"  random_entry    bench_sharpe_p95={rand['bench_sharpe_p95']:.2f}")

    # Final verdict
    real_sharpe = res.get("sharpe", 0.0)
    robust = (
        res["pass"]
        and sf["pass"]
        and oos["pass"]
        and pass_thresh >= len(bt_thresh) * 0.75
        and real_sharpe > shuf["null_sharpe_p95"]
        and real_sharpe > rand["bench_sharpe_p95"]
    )

    verdict = "GO" if robust else "NO-GO"
    print(f"\n{'='*40}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*40}")
    if not robust:
        if not res["pass"]:
            print("  Failures:", res.get("failures", []))
        if not sf["pass"]:
            print("  Fee stress failed")
        if not oos["pass"]:
            print("  OOS failed")
        if pass_thresh < len(bt_thresh) * 0.75:
            print(f"  Threshold sweep: only {pass_thresh}/{len(bt_thresh)} pass")
        if real_sharpe <= shuf["null_sharpe_p95"]:
            print(f"  Sharpe below shuffle null distribution")
        if real_sharpe <= rand["bench_sharpe_p95"]:
            print(f"  Sharpe below random entry benchmark")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StatArb BTC/ETH backtest orchestrator")
    parser.add_argument("--fetch", action="store_true", help="Fetch fresh data from Hyperliquid")
    parser.add_argument("--testnet", action="store_true", help="Use Hyperliquid testnet endpoint")
    args = parser.parse_args()
    run(fetch=args.fetch, testnet=args.testnet)
