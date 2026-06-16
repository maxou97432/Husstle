"""
Multi-pair StatArb with OOS pair selection (proper).

The brief calls for "local mean-reversion, NOT eternal cointegration": we must
*select* which pairs to trade, not trade all blindly. Doing it on the full
sample is selection bias. Doing it in real time alone (per-bar ADF) doesn't
help: short windows of apparent cointegration generate losing trades before
the kill-switch fires.

Proper procedure:
  1. Split data: SELECTION_FRAC (default 35%) = training window.
  2. For each pair, compute the ADF p-value of the spread over the training
     window only.
  3. Keep pairs whose training ADF p < ADF_SELECT (default 0.10).
  4. Backtest those pairs ONLY on the held-out window.
  5. Aggregate equal-weight; report portfolio Sharpe + bootstrap CI.

This is the §6 gate #3 (out-of-sample) applied to pair selection.
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
from statsmodels.tsa.stattools import adfuller
from backtest.engine import BacktestConfig, run_backtest
from backtest.metrics import (
    sharpe, max_drawdown, win_rate, profit_factor, bootstrap_sharpe_ci,
)


def _load_universe():
    conn = get_conn()
    out = {}
    for coin in UNIVERSE:
        c = load_candles_pl(conn, coin).sort("ts").collect()
        f = load_funding_pl(conn, coin).sort("ts").collect()
        if c.height == 0:
            continue
        out[coin] = {
            "ts": c["ts"].to_numpy(),
            "close": c["close"].to_numpy(),
            "funding": dict(zip(f["ts"].to_list(), f["rate"].to_list())) if f.height else {},
        }
    conn.close()
    return out


def _aligned(y_data, x_data):
    common = np.intersect1d(y_data["ts"], x_data["ts"])
    if len(common) < HEDGE_WINDOW + Z_WINDOW + 100:
        return None
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


def _adf_pvalue_on_window(y_close, x_close, start, end) -> float:
    """ADF on the spread computed via causal rolling OLS up to `end`."""
    log_y = np.log(y_close[:end])
    log_x = np.log(x_close[:end])
    spread, _, _ = compute_spread(log_y, log_x, HEDGE_WINDOW)
    window_data = spread[start:end]
    window_data = window_data[~np.isnan(window_data)]
    if len(window_data) < 50:
        return 1.0
    try:
        return float(adfuller(window_data, autolag="AIC")[1])
    except Exception:
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=float, default=1.0)
    ap.add_argument("--exit",  type=float, default=0.25)
    ap.add_argument("--sel-frac", type=float, default=0.35,
                    help="Fraction of data used for pair selection (default 0.35)")
    ap.add_argument("--adf-select", type=float, default=0.10,
                    help="Max ADF p-value to KEEP a pair (default 0.10)")
    args = ap.parse_args()

    cfg = BacktestConfig(entry_z=args.entry, exit_z=args.exit)
    print(f"Multi-pair OOS  TF={INTERVAL}  entry={cfg.entry_z}  exit={cfg.exit_z}  "
          f"sel_frac={args.sel_frac}  adf_select={args.adf_select}\n")

    uni = _load_universe()
    pairs = list(itertools.combinations(uni.keys(), 2))
    print(f"Universe: {list(uni.keys())}   pairs={len(pairs)}\n")

    # ── Pass 1 : selection on training window ────────────────────────────
    pair_data = {}
    sel_results = []
    for y, x in pairs:
        ali = _aligned(uni[y], uni[x])
        if ali is None:
            continue
        pair_data[(y, x)] = ali
        n = len(ali[4])
        cut = int(n * args.sel_frac)
        # Selection uses bars [HEDGE_WINDOW, cut] of the spread series (training only).
        pval = _adf_pvalue_on_window(ali[0], ali[1], HEDGE_WINDOW, cut)
        sel_results.append({"pair": f"{y}/{x}", "n": n, "cut": cut, "adf_train": pval,
                            "kept": pval < args.adf_select})

    print("Pair selection (training window):")
    print(f"  {'pair':>10}  {'cut_bar':>7}  {'ADF_p_train':>12}  {'kept':>4}")
    for r in sorted(sel_results, key=lambda r: r["adf_train"]):
        print(f"  {r['pair']:>10}  {r['cut']:>7}  {r['adf_train']:>12.4f}   "
              f"{'YES' if r['kept'] else 'no':>4}")

    kept_pairs = [r["pair"] for r in sel_results if r["kept"]]
    print(f"\n  Kept: {len(kept_pairs)}/{len(sel_results)} pairs.\n")

    if not kept_pairs:
        print("No pair survived the selection filter. Try --adf-select 0.20.")
        return

    # ── Pass 2 : backtest kept pairs OOS only ────────────────────────────
    rows = []
    pair_equity = {}
    for r in sel_results:
        if not r["kept"]:
            continue
        y, x = r["pair"].split("/")
        cy, cx, fy, fx, ts = pair_data[(y, x)]
        # OOS slice: from `cut` onwards. Re-compute signals on full series so β
        # stays causal, then trim to OOS region for backtest.
        log_y = np.log(cy); log_x = np.log(cx)
        spread, beta, _ = compute_spread(log_y, log_x, HEDGE_WINDOW)
        z = rolling_zscore(spread, Z_WINDOW)
        adf = rolling_adf_pvalue(spread, ADF_WINDOW, ADF_STEP)
        cut = r["cut"]
        trades, equity = run_backtest(
            cy[cut:], cx[cut:], spread[cut:], z[cut:], adf[cut:],
            fy[cut:], fx[cut:], beta[cut:], cfg,
        )
        rows.append({
            "pair": r["pair"],
            "oos_bars": len(cy) - cut,
            "n_trades": len(trades),
            "sharpe": sharpe(equity, bars_per_year=BARS_PER_YEAR),
            "wr": win_rate(trades),
            "pf": profit_factor(trades),
            "pnl": float(equity[-1]),
            "mdd": max_drawdown(equity),
            "adf_train": r["adf_train"],
        })
        pair_equity[(y, x)] = (ts[cut:], equity)

    print("OOS backtest of kept pairs:")
    print(f"  {'pair':>10}  {'OOSbars':>7}  {'trades':>6}  {'Sharpe':>7}  "
          f"{'WR':>6}  {'PF':>6}  {'netPnL':>8}  {'maxDD':>8}  {'ADFtr':>7}")
    print("  " + "-" * 84)
    for r in sorted(rows, key=lambda r: -r["sharpe"]):
        pf = f"{r['pf']:.2f}" if np.isfinite(r["pf"]) else "  inf"
        print(f"  {r['pair']:>10}  {r['oos_bars']:>7d}  {r['n_trades']:>6d}  "
              f"{r['sharpe']:>+7.2f}  {r['wr']:>6.1%}  {pf:>6}  "
              f"{r['pnl']:>+8.4f}  {r['mdd']:>+8.4f}  {r['adf_train']:>7.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────
    master = sorted(set().union(*[set(ts.tolist()) for ts, _ in pair_equity.values()]))
    idx_master = {t: i for i, t in enumerate(master)}
    portfolio = np.zeros(len(master))
    for ts, equity in pair_equity.values():
        incr = np.diff(equity, prepend=0.0)
        for k, t in enumerate(ts):
            portfolio[idx_master[int(t)]] += incr[k]
    portfolio /= len(pair_equity)
    port_eq = np.cumsum(portfolio)

    n_total = sum(r["n_trades"] for r in rows)
    port_sharpe = sharpe(port_eq, bars_per_year=BARS_PER_YEAR)
    port_mdd = max_drawdown(port_eq)
    boot_lo, boot_hi = bootstrap_sharpe_ci(port_eq, n_boot=2000, ci=0.95)

    print("\n" + "=" * 60)
    print(f"  OOS PORTFOLIO  ({len(pair_equity)} pairs, equal-weight)")
    print("=" * 60)
    print(f"  Total OOS trades : {n_total}")
    print(f"  Sharpe (net)     : {port_sharpe:+.2f}")
    print(f"  Bootstrap 95% CI : [{boot_lo:+.2f}, {boot_hi:+.2f}]")
    print(f"  Net PnL          : {port_eq[-1]:+.4f}")
    print(f"  Max drawdown     : {port_mdd:+.4f}")
    print("=" * 60)
    if n_total >= 100:
        print("  Power gate (n>=100)         : PASS")
    else:
        print(f"  Power gate (n>=100)         : FAIL ({n_total})")
    if boot_lo > 0:
        print("  Bootstrap CI lower > 0      : PASS")
    else:
        print("  Bootstrap CI lower > 0      : FAIL")


if __name__ == "__main__":
    main()
