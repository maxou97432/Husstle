"""
Threshold sweep — grid search over (entry_z, exit_z) on real Hyperliquid data.

Filtering rule from the brief (§6, power): only cells with n_trades >= 100 are
considered statistically meaningful. Below that we mark them but do NOT call
them an edge.

Output: a heatmap-style table for Sharpe, n_trades, win_rate, profit_factor.
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal.spread import compute_spread, rolling_zscore
from signal.cointegration import rolling_adf_pvalue
from backtest.engine import BacktestConfig, run_backtest
from backtest.metrics import sharpe, win_rate, profit_factor
from tests._data import get_data

HEDGE_W = 240
Z_W = 60
ADF_W = 240
ADF_STEP = 6

ENTRY_GRID = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50]
EXIT_GRID  = [0.00, 0.25, 0.50, 0.75]
MIN_TRADES = 100

# Larger figures so the table reads cleanly
def _f(x, w=7, d=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return f"{'  --  ':>{w}}"
    return f"{x:>{w}.{d}f}"


def main() -> None:
    data, source = get_data(prefer_real=True)
    print(f"Data source: {source}\n")
    close_eth, close_btc, fund_eth, fund_btc = data

    log_eth = np.log(close_eth)
    log_btc = np.log(close_btc)
    spread, beta, _ = compute_spread(log_eth, log_btc, HEDGE_W)
    z = rolling_zscore(spread, Z_W)
    adf = rolling_adf_pvalue(spread, ADF_W, ADF_STEP)

    # Pre-allocate matrices
    M_sharpe = np.full((len(ENTRY_GRID), len(EXIT_GRID)), np.nan)
    M_trades = np.zeros((len(ENTRY_GRID), len(EXIT_GRID)), dtype=int)
    M_wr     = np.full_like(M_sharpe, np.nan)
    M_pf     = np.full_like(M_sharpe, np.nan)
    M_pnl    = np.full_like(M_sharpe, np.nan)
    best = (-1e9, None)

    for i, ez in enumerate(ENTRY_GRID):
        for j, xz in enumerate(EXIT_GRID):
            cfg = BacktestConfig(entry_z=ez, exit_z=xz)
            trades, equity = run_backtest(
                close_eth, close_btc, spread, z, adf,
                fund_eth, fund_btc, beta, cfg,
            )
            M_trades[i, j] = len(trades)
            if len(trades) == 0:
                continue
            M_sharpe[i, j] = sharpe(equity)
            M_wr[i, j]     = win_rate(trades)
            M_pf[i, j]     = profit_factor(trades)
            M_pnl[i, j]    = float(equity[-1])
            if M_trades[i, j] >= MIN_TRADES and M_sharpe[i, j] > best[0]:
                best = (M_sharpe[i, j], (ez, xz))

    # ── Print tables ────────────────────────────────────────────────────────
    def header():
        return "  entry_z \\ exit_z |" + "".join(f"{xz:>8.2f}" for xz in EXIT_GRID)

    def print_table(title: str, M: np.ndarray, fmt, fill_eligible_mark: bool = False):
        print(f"\n{title}")
        print(header())
        print("  " + "-" * (len(header()) - 2))
        for i, ez in enumerate(ENTRY_GRID):
            row = f"  {ez:>14.2f}  |"
            for j in range(len(EXIT_GRID)):
                v = M[i, j]
                txt = fmt(v)
                if fill_eligible_mark and M_trades[i, j] >= MIN_TRADES:
                    txt = f"*{txt.strip():>6}"
                row += f"{txt:>8}"
            print(row)

    print_table("n_trades  (cells with * are eligible: n_trades >= {})".format(MIN_TRADES),
                M_trades, lambda v: f"{int(v):>7d}", fill_eligible_mark=True)
    print_table("Sharpe (annualised, net of fees+funding)",
                M_sharpe, lambda v: _f(v, 7, 2))
    print_table("Win rate",
                M_wr, lambda v: _f(v, 7, 3))
    print_table("Profit factor",
                M_pf, lambda v: _f(v, 7, 2))
    print_table("Final net PnL (fractional)",
                M_pnl, lambda v: _f(v, 7, 4))

    # ── Summary ────────────────────────────────────────────────────────────
    eligible = M_trades >= MIN_TRADES
    n_elig = int(eligible.sum())
    print("\n" + "=" * 60)
    print(f"  Eligible cells (n_trades >= {MIN_TRADES}): {n_elig} / {M_trades.size}")
    if best[1] is not None:
        ez, xz = best[1]
        i = ENTRY_GRID.index(ez); j = EXIT_GRID.index(xz)
        print(f"  Best eligible cell: entry_z={ez:.2f}, exit_z={xz:.2f}")
        print(f"    Sharpe={M_sharpe[i,j]:+.2f}  n_trades={M_trades[i,j]}  "
              f"win_rate={M_wr[i,j]:.1%}  PF={M_pf[i,j]:.2f}  netPnL={M_pnl[i,j]:+.4f}")
    else:
        print("  No eligible cell. Conclusions:")
        print("   - At 4h timeframe, the BTC/ETH spread does not generate enough")
        print("     mean-reversion events on the past year to reach statistical power.")
        print("   - Next levers (brief §6): drop to 1h TF, or extend lookback to 2-3y.")
    print("=" * 60)


if __name__ == "__main__":
    main()
