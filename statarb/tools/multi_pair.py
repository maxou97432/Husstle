"""
Multi-pair StatArb runner.

For every unordered pair (y, x) in UNIVERSE with y != x:
  - load aligned candles + funding
  - compute causal β/α, spread, z-score, ADF
  - run the same backtest engine (same fees, leverage, kill-switch)
At the end, aggregate the per-pair PnL into a portfolio equity curve
under equal-weight capital allocation (1/N per pair), so the total
gross leverage matches the brief (4× per pair × 1/N pairs <= 4×).

Reports per-pair stats AND the aggregated portfolio stats with the
proper bars-per-year annualisation and bootstrap CI.
"""
from __future__ import annotations

import os
import sys
import argparse
import itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    UNIVERSE, HEDGE_WINDOW, Z_WINDOW, ADF_WINDOW, ADF_STEP,
    BARS_PER_YEAR, INTERVAL, INTERVAL_MS,
)
from data.store import get_conn, load_candles_pl, load_funding_pl
from signal.spread import compute_spread, rolling_zscore
from signal.cointegration import rolling_adf_pvalue
from backtest.engine import BacktestConfig, run_backtest
from backtest.metrics import (
    sharpe, max_drawdown, win_rate, profit_factor, bootstrap_sharpe_ci,
)


def _load_universe() -> dict:
    """Returns {coin: {'ts': np.ndarray, 'close': np.ndarray, 'funding': dict[int, float]}}."""
    conn = get_conn()
    out = {}
    for coin in UNIVERSE:
        c = load_candles_pl(conn, coin).sort("ts").collect()
        f = load_funding_pl(conn, coin).sort("ts").collect()
        if c.height == 0:
            print(f"  skip {coin}: no candles")
            continue
        out[coin] = {
            "ts": c["ts"].to_numpy(),
            "close": c["close"].to_numpy(),
            "funding": dict(zip(f["ts"].to_list(), f["rate"].to_list())) if f.height else {},
        }
    conn.close()
    return out


def _aligned(y_data: dict, x_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inner-join two coins on ts; aggregate hourly funding to TF window."""
    common = np.intersect1d(y_data["ts"], x_data["ts"])
    if len(common) < HEDGE_WINDOW + Z_WINDOW + 50:
        return None  # not enough overlap
    y_map = dict(zip(y_data["ts"].tolist(), y_data["close"].tolist()))
    x_map = dict(zip(x_data["ts"].tolist(), x_data["close"].tolist()))
    cy = np.array([y_map[t] for t in common])
    cx = np.array([x_map[t] for t in common])

    def agg(fmap):
        if not fmap:
            return np.zeros(len(common))
        f_ts = np.array(sorted(fmap), dtype=np.int64)
        f_rate = np.array([fmap[t] for t in sorted(fmap)])
        out = np.zeros(len(common))
        for i, t in enumerate(common):
            m = (f_ts > t - INTERVAL_MS) & (f_ts <= t)
            out[i] = f_rate[m].sum() if m.any() else 0.0
        return out

    return cy, cx, agg(y_data["funding"]), agg(x_data["funding"]), common


def run_pair(y_close, x_close, fy, fx, cfg) -> tuple:
    log_y = np.log(y_close); log_x = np.log(x_close)
    spread, beta, _ = compute_spread(log_y, log_x, HEDGE_WINDOW)
    z = rolling_zscore(spread, Z_WINDOW)
    adf = rolling_adf_pvalue(spread, ADF_WINDOW, ADF_STEP)
    trades, equity = run_backtest(y_close, x_close, spread, z, adf, fy, fx, beta, cfg)
    return trades, equity, spread, z, adf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=float, default=1.0, help="entry_z")
    ap.add_argument("--exit",  type=float, default=0.25, help="exit_z")
    args = ap.parse_args()

    cfg = BacktestConfig(entry_z=args.entry, exit_z=args.exit)
    print(f"Multi-pair StatArb  TF={INTERVAL}  entry_z={cfg.entry_z}  exit_z={cfg.exit_z}\n")

    uni = _load_universe()
    pairs = list(itertools.combinations(uni.keys(), 2))
    print(f"Universe: {list(uni.keys())}   ({len(pairs)} unordered pairs)\n")

    rows = []
    pair_equity = {}     # pair -> aligned equity series (same length, 0 outside)
    union_ts = None

    # First pass: union of all common timestamps across pairs (use shortest pair)
    pair_data = {}
    for y, x in pairs:
        ali = _aligned(uni[y], uni[x])
        if ali is None:
            print(f"  {y}/{x}: insufficient overlap, skip")
            continue
        pair_data[(y, x)] = ali
        if union_ts is None or len(ali[4]) > len(union_ts):
            union_ts = ali[4]   # use longest series as the master timeline

    # Backtest each pair
    for (y, x), (cy, cx, fy, fx, ts) in pair_data.items():
        trades, equity, _, _, _ = run_pair(cy, cx, fy, fx, cfg)
        rows.append({
            "pair": f"{y}/{x}",
            "bars": len(cy),
            "n_trades": len(trades),
            "sharpe": sharpe(equity, bars_per_year=BARS_PER_YEAR),
            "wr": win_rate(trades),
            "pf": profit_factor(trades),
            "pnl": float(equity[-1]),
            "mdd": max_drawdown(equity),
        })
        pair_equity[(y, x)] = (ts, equity)

    # ── Per-pair table ───────────────────────────────────────────────────
    print(f"{'pair':>10}  {'bars':>5}  {'trades':>6}  {'Sharpe':>7}  {'WR':>6}  {'PF':>6}  {'netPnL':>8}  {'maxDD':>8}")
    print("  " + "-" * 70)
    for r in sorted(rows, key=lambda r: -r["sharpe"]):
        pf = f"{r['pf']:.2f}" if np.isfinite(r["pf"]) else "  inf"
        print(f"{r['pair']:>10}  {r['bars']:>5d}  {r['n_trades']:>6d}  "
              f"{r['sharpe']:>+7.2f}  {r['wr']:>6.1%}  {pf:>6}  "
              f"{r['pnl']:>+8.4f}  {r['mdd']:>+8.4f}")

    # ── Aggregate (equal-weight) ─────────────────────────────────────────
    # Align all per-pair equity curves to a common timestamp axis.
    if not pair_equity:
        print("\nNo pairs to aggregate.")
        return
    master = sorted(set().union(*[set(ts.tolist()) for ts, _ in pair_equity.values()]))
    idx_master = {t: i for i, t in enumerate(master)}
    portfolio = np.zeros(len(master))
    for ts, equity in pair_equity.values():
        # equity is cumulative; convert to per-bar PnL increments then place on master
        incr = np.diff(equity, prepend=0.0)
        for k, t in enumerate(ts):
            portfolio[idx_master[int(t)]] += incr[k]
    portfolio /= len(pair_equity)       # equal-weight capital allocation
    port_eq = np.cumsum(portfolio)

    n_total = sum(r["n_trades"] for r in rows)
    port_sharpe = sharpe(port_eq, bars_per_year=BARS_PER_YEAR)
    port_mdd = max_drawdown(port_eq)
    boot_lo, boot_hi = bootstrap_sharpe_ci(port_eq, n_boot=2000, ci=0.95)

    print("\n" + "=" * 60)
    print("  PORTFOLIO (equal-weight, 1/{} per pair)".format(len(pair_equity)))
    print("=" * 60)
    print(f"  Pairs traded     : {len(pair_equity)}")
    print(f"  Total trades     : {n_total}")
    print(f"  Sharpe (net)     : {port_sharpe:+.2f}")
    print(f"  Bootstrap 95% CI : [{boot_lo:+.2f}, {boot_hi:+.2f}]")
    print(f"  Net PnL          : {port_eq[-1]:+.4f}")
    print(f"  Max drawdown     : {port_mdd:+.4f}")
    print("=" * 60)
    if n_total >= 100:
        print("  Power gate (n>=100): PASS")
    else:
        print(f"  Power gate (n>=100): FAIL ({n_total} trades — still under threshold)")
    if boot_lo > 0:
        print("  Bootstrap lower bound > 0: PASS (edge survives sampling)")
    else:
        print("  Bootstrap lower bound <= 0: FAIL")


if __name__ == "__main__":
    main()
