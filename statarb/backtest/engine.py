from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class ExitReason(str, Enum):
    SIGNAL = "signal"
    STOP = "stop"
    KILL_SWITCH = "kill_switch"


@dataclass
class BacktestConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5        # mean-reversion target
    stop_z: float = 3.5        # hard stop
    adf_entry_max: float = 0.05   # required p-value for new entries
    adf_kill_max: float = 0.10    # kill-switch: flat existing position above this
    leverage: float = 2.0         # per leg (~4× gross)
    fee_maker: float = 0.0002     # normal entry/exit
    fee_taker: float = 0.0005     # stop & kill-switch exits
    slippage: float = 0.0002      # per fill
    fee_stress: float = 1.0       # multiplier applied to all fees
    candle_interval_h: float = 4.0
    funding_interval_h: float = 1.0


@dataclass
class Trade:
    entry_bar: int
    direction: int                # +1 long spread / -1 short spread
    entry_spread: float
    exit_bar: int = -1
    exit_spread: float = 0.0
    exit_reason: ExitReason = ExitReason.SIGNAL
    pnl: float = 0.0
    funding_paid: float = 0.0    # net funding cost (positive = cost)
    fees_paid: float = 0.0


def _fee(cfg: BacktestConfig, price_y: float, price_x: float, beta: float, taker: bool) -> float:
    rate = cfg.fee_taker if taker else cfg.fee_maker
    # fees on both legs, levered notional
    return (rate + cfg.slippage) * cfg.fee_stress * cfg.leverage * (abs(price_y) + abs(beta * price_x))


def run_backtest(
    close_y: np.ndarray,
    close_x: np.ndarray,
    spread: np.ndarray,
    zscore: np.ndarray,
    adf_pval: np.ndarray,
    funding_y: np.ndarray | None,
    funding_x: np.ndarray | None,
    hedge_beta: np.ndarray,
    cfg: BacktestConfig | None = None,
) -> tuple[list[Trade], np.ndarray]:
    """
    Signal at close[i] → execution at open[i+1].
    PnL = direction * Δspread * leverage (log-spread ≈ % return, levered).
    Funding applied per bar held, on both legs, realised rate.
    Maker fees for normal entry/exit; taker for stop & kill-switch.
    Returns (trades, equity_curve).
    """
    if cfg is None:
        cfg = BacktestConfig()

    n = len(close_y)
    equity = np.zeros(n)
    trades: list[Trade] = []
    position: Trade | None = None
    cumulative_pnl = 0.0

    for i in range(1, n - 1):
        z = zscore[i]
        p = adf_pval[i]
        beta = hedge_beta[i]

        if np.isnan(z) or np.isnan(beta):
            equity[i] = cumulative_pnl
            continue

        exec_bar = i + 1
        py = close_y[exec_bar]
        px = close_x[exec_bar]

        # ── Handle open position ──────────────────────────────────────────
        if position is not None:
            abs_z = abs(z)
            kill = (not np.isnan(p)) and (p > cfg.adf_kill_max)
            stopped = abs_z > cfg.stop_z
            signal_exit = (
                (position.direction == +1 and z > -cfg.exit_z)
                or (position.direction == -1 and z < cfg.exit_z)
            )

            if kill or stopped or signal_exit:
                reason = (
                    ExitReason.KILL_SWITCH if kill
                    else ExitReason.STOP if stopped
                    else ExitReason.SIGNAL
                )
                is_urgent = reason in (ExitReason.STOP, ExitReason.KILL_SWITCH)
                position.exit_bar = exec_bar
                position.exit_spread = spread[exec_bar]
                position.fees_paid += _fee(cfg, py, px, beta, taker=is_urgent)
                raw_pnl = position.direction * (position.exit_spread - position.entry_spread) * cfg.leverage
                position.pnl = raw_pnl - position.fees_paid - position.funding_paid
                position.exit_reason = reason
                cumulative_pnl += position.pnl
                trades.append(position)
                position = None

            else:
                # Accrue funding every candle — funding_y/x are 4h-aggregated rates
                if funding_y is not None and funding_x is not None:
                    fy = funding_y[i] if i < len(funding_y) else 0.0
                    fx = funding_x[i] if i < len(funding_x) else 0.0
                    # long spread: long y, short x → pay fy, receive fx (sign depends on direction)
                    net = position.direction * (fy * close_y[i] - fx * close_x[i] * beta) * cfg.leverage
                    position.funding_paid += net  # positive = cost to position

            equity[i] = cumulative_pnl
            continue

        # ── Open new position ─────────────────────────────────────────────
        cointegrated = (not np.isnan(p)) and (p < cfg.adf_entry_max)
        if cointegrated and not np.isnan(z):
            direction = None
            if z < -cfg.entry_z:
                direction = +1   # long spread
            elif z > cfg.entry_z:
                direction = -1   # short spread

            if direction is not None:
                position = Trade(
                    entry_bar=exec_bar,
                    direction=direction,
                    entry_spread=spread[exec_bar],
                )
                position.fees_paid = _fee(cfg, py, px, beta, taker=False)

        equity[i] = cumulative_pnl

    return trades, equity
