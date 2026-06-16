from __future__ import annotations

import numpy as np
from core.backtest.engine import Trade

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


def breakeven_win_rate(trades: list[Trade]) -> float:
    """
    Minimum win rate to break even given the average win/loss sizes.
    BEWR = avg_loss / (avg_win + avg_loss)
    """
    winners = [t.pnl for t in trades if t.pnl > 0]
    losers = [abs(t.pnl) for t in trades if t.pnl < 0]
    if not winners or not losers:
        return float("nan")
    avg_win = float(np.mean(winners))
    avg_loss = float(np.mean(losers))
    return avg_loss / (avg_win + avg_loss)


def bootstrap_sharpe_ci(
    equity: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap CI on the annualised Sharpe by resampling returns.
    Decision is based on the lower CI bound (§6 of brief).
    """
    returns = np.diff(equity)
    if len(returns) < 2:
        return (0.0, 0.0)
    boot_sharpes = []
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        std = sample.std()
        if std == 0:
            boot_sharpes.append(0.0)
            continue
        boot_sharpes.append(float(sample.mean() / std * np.sqrt(ANNUAL_BARS_4H)))
    lo = float(np.percentile(boot_sharpes, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_sharpes, (1 + ci) / 2 * 100))
    return lo, hi


GATES = {
    "min_trades": 30,
    "min_sharpe": 0.5,
    "min_win_rate": 0.45,
    "min_profit_factor": 1.1,
    "bootstrap_sharpe_lo": 0.0,   # lower CI bound must exceed 0
    "max_bewr_margin": 0.05,       # actual win_rate > bewr + this margin
}


def evaluate(trades: list[Trade], equity: np.ndarray) -> dict:
    if not trades:
        return {"pass": False, "failures": ["no trades"]}

    s = sharpe(equity)
    mdd = max_drawdown(equity)
    wr = win_rate(trades)
    pf = profit_factor(trades)
    bewr = breakeven_win_rate(trades)
    boot_lo, boot_hi = bootstrap_sharpe_ci(equity)

    results = {
        "n_trades": len(trades),
        "sharpe": round(s, 3),
        "max_drawdown": round(mdd, 4),
        "win_rate": round(wr, 3),
        "profit_factor": round(pf, 3),
        "breakeven_win_rate": round(bewr, 3) if not np.isnan(bewr) else None,
        "bootstrap_sharpe_ci": (round(boot_lo, 3), round(boot_hi, 3)),
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
        failures.append(f"bootstrap_sharpe_lo={boot_lo:.3f} < 0")
    if not np.isnan(bewr) and wr < bewr + GATES["max_bewr_margin"]:
        failures.append(f"win_rate={wr:.2%} not > breakeven={bewr:.2%} + margin")

    results["pass"] = len(failures) == 0
    results["failures"] = failures
    return results
