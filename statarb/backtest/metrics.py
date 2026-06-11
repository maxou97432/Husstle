from __future__ import annotations

import numpy as np
from scipy import stats
from backtest.engine import Trade


ANNUAL_BARS_4H = 365 * 6  # 4h candles per year


def sharpe(equity: np.ndarray, bars_per_year: int = ANNUAL_BARS_4H) -> float:
    returns = np.diff(equity)
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def win_rate(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    return sum(1 for t in trades if t.pnl > 0) / len(trades)


def profit_factor(trades: list[Trade]) -> float:
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss == 0:
        return float("inf")
    return gross_win / gross_loss


def bootstrap_ci(
    trades: list[Trade],
    stat_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    pnls = np.array([t.pnl for t in trades])
    if len(pnls) == 0:
        return (0.0, 0.0)
    samples = np.random.choice(pnls, size=(n_boot, len(pnls)), replace=True)
    stats_boot = np.array([stat_fn(s) for s in samples])
    lo = np.percentile(stats_boot, (1 - ci) / 2 * 100)
    hi = np.percentile(stats_boot, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


def _mean(arr: np.ndarray) -> float:
    return float(arr.mean())


GATES = {
    "min_trades": 30,
    "min_sharpe": 0.5,
    "max_drawdown_pct": -0.20,   # relative to peak equity (if equity is %-based)
    "min_win_rate": 0.45,
    "min_profit_factor": 1.1,
    "bootstrap_sharpe_lo": 0.0,  # lower CI bound must be > 0
}


def evaluate(
    trades: list[Trade],
    equity: np.ndarray,
) -> dict:
    if not trades:
        return {"pass": False, "reason": "no trades"}

    s = sharpe(equity)
    mdd = max_drawdown(equity)
    wr = win_rate(trades)
    pf = profit_factor(trades)
    pnls = np.array([t.pnl for t in trades])
    boot_lo, boot_hi = bootstrap_ci(trades, _mean)

    results = {
        "n_trades": len(trades),
        "sharpe": s,
        "max_drawdown": mdd,
        "win_rate": wr,
        "profit_factor": pf,
        "bootstrap_mean_ci": (boot_lo, boot_hi),
    }

    failures = []
    if len(trades) < GATES["min_trades"]:
        failures.append(f"n_trades={len(trades)} < {GATES['min_trades']}")
    if s < GATES["min_sharpe"]:
        failures.append(f"sharpe={s:.2f} < {GATES['min_sharpe']}")
    if wr < GATES["min_win_rate"]:
        failures.append(f"win_rate={wr:.2%} < {GATES['min_win_rate']:.0%}")
    if pf < GATES["min_profit_factor"]:
        failures.append(f"profit_factor={pf:.2f} < {GATES['min_profit_factor']}")
    if boot_lo < GATES["bootstrap_sharpe_lo"]:
        failures.append(f"bootstrap_ci_lo={boot_lo:.4f} < 0")

    results["pass"] = len(failures) == 0
    results["failures"] = failures
    return results
