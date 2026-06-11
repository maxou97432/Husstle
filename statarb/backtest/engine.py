from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    entry_z: float = 2.0
    exit_z: float = 0.0
    adf_pval_max: float = 0.05
    leverage: float = 1.0
    fee_maker: float = 0.0002
    fee_taker: float = 0.0005
    slippage: float = 0.0002
    fee_stress: float = 1.0  # multiplier applied to all fees
    candle_interval_h: float = 4.0
    funding_interval_h: float = 1.0


@dataclass
class Trade:
    entry_bar: int
    direction: int        # +1 long spread / -1 short spread
    entry_spread: float
    exit_bar: int = -1
    exit_spread: float = 0.0
    pnl: float = 0.0
    funding_paid: float = 0.0
    fees_paid: float = 0.0


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
    Signal observed at close[i], execution at open[i+1].
    Returns (trades, equity_curve).
    funding_y / funding_x are 4h-aggregated funding rates aligned to candle timestamps.
    """
    if cfg is None:
        cfg = BacktestConfig()

    n = len(close_y)
    equity = np.zeros(n)
    trades: list[Trade] = []
    position: Trade | None = None

    fee_open = (cfg.fee_taker + cfg.slippage) * cfg.fee_stress
    fee_close = (cfg.fee_taker + cfg.slippage) * cfg.fee_stress

    for i in range(1, n - 1):
        z = zscore[i]
        p = adf_pval[i]
        beta = hedge_beta[i]

        if np.isnan(z) or np.isnan(p) or np.isnan(beta):
            equity[i] = equity[i - 1]
            continue

        exec_bar = i + 1
        entry_price_y = close_y[exec_bar]
        entry_price_x = close_x[exec_bar]

        if position is None:
            cointegrated = p < cfg.adf_pval_max
            if cointegrated and z > cfg.entry_z:
                position = Trade(
                    entry_bar=exec_bar,
                    direction=-1,
                    entry_spread=spread[exec_bar],
                )
                position.fees_paid += fee_open * (abs(entry_price_y) + abs(beta * entry_price_x))
            elif cointegrated and z < -cfg.entry_z:
                position = Trade(
                    entry_bar=exec_bar,
                    direction=+1,
                    entry_spread=spread[exec_bar],
                )
                position.fees_paid += fee_open * (abs(entry_price_y) + abs(beta * entry_price_x))
        else:
            should_exit = (
                (position.direction == -1 and z < cfg.exit_z)
                or (position.direction == +1 and z > -cfg.exit_z)
                or np.isnan(z)
            )
            if should_exit:
                position.exit_bar = exec_bar
                position.exit_spread = spread[exec_bar]
                raw_pnl = position.direction * (position.exit_spread - position.entry_spread)
                position.fees_paid += fee_close * (abs(close_y[exec_bar]) + abs(beta * close_x[exec_bar]))
                position.pnl = raw_pnl - position.fees_paid - position.funding_paid
                trades.append(position)
                position = None

            elif funding_y is not None and funding_x is not None:
                candles_per_funding = int(cfg.funding_interval_h / cfg.candle_interval_h * 4)
                if i % max(1, candles_per_funding) == 0:
                    fy = funding_y[i] if i < len(funding_y) else 0.0
                    fx = funding_x[i] if i < len(funding_x) else 0.0
                    # long spread: long y short x
                    net_funding = position.direction * (fy * close_y[i] - fx * close_x[i] * beta)
                    position.funding_paid -= net_funding  # positive = cost

        unrealised = 0.0
        if position is not None:
            unrealised = position.direction * (spread[i] - position.entry_spread) - position.fees_paid
        equity[i] = (equity[i - 1] if i > 0 else 0.0) + (
            trades[-1].pnl if trades and trades[-1].exit_bar == exec_bar else 0.0
        )

    return trades, equity
