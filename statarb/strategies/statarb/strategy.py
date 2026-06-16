"""
StatArb strategy wrapped as a Strategy implementation for the orchestrator.

Captures the same logic that lived in live_bot.py:
  - Compute β/α, spread, z-score, ADF causally from recent candles.
  - Enter on |z|>entry_z + ADF p<adf_entry_max, dollar-neutral 2-leg.
  - Exit on signal mean-reversion, stop, or ADF kill.

This class knows NOTHING about Hyperliquid; it only returns Decision objects.
The orchestrator translates Decisions into orders.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from strategies.base import Strategy, Decision, Action, Leg
from strategies.statarb.config import (
    UNIVERSE, HEDGE_WINDOW, Z_WINDOW, ADF_WINDOW, ADF_STEP,
    ENTRY_Z, EXIT_Z, STOP_Z, ADF_ENTRY_MAX, ADF_KILL_MAX,
)
from strategies.statarb.spread import compute_spread, rolling_zscore
from strategies.statarb.cointegration import rolling_adf_pvalue
from core.risk.sizing import leg_sizes


@dataclass
class StatArbParams:
    pair_y: str = "ETH"           # spread = log(Y) − β·log(X) − α
    pair_x: str = "BTC"
    entry_z: float = ENTRY_Z
    exit_z: float = EXIT_Z
    stop_z: float = STOP_Z
    adf_entry_max: float = ADF_ENTRY_MAX
    adf_kill_max: float = ADF_KILL_MAX
    capital_usd: float = 100.0
    leverage: int = 2
    safety_stop_pct: float = 0.05


class StatArbStrategy(Strategy):
    def __init__(self, params: StatArbParams | None = None):
        self.p = params or StatArbParams()

    @property
    def name(self) -> str:
        return f"statarb-{self.p.pair_y}-{self.p.pair_x}"

    def required_coins(self) -> list[str]:
        return [self.p.pair_y, self.p.pair_x]

    def compute_signal(self, market_data: dict, position: dict | None) -> Decision:
        """
        market_data must contain {pair_y: {close: np.ndarray, ts: np.ndarray, top_bid, top_ask},
                                  pair_x: {...}}
        position: None if flat, else dict returned at previous ENTER.
        """
        y_data = market_data[self.p.pair_y]
        x_data = market_data[self.p.pair_x]
        py_close = y_data["close"]
        px_close = x_data["close"]

        if len(py_close) < HEDGE_WINDOW + Z_WINDOW + 10:
            return Decision(action=Action.HOLD, reason="warmup")

        log_y = np.log(py_close)
        log_x = np.log(px_close)
        spread, beta, alpha = compute_spread(log_y, log_x, HEDGE_WINDOW)
        z = rolling_zscore(spread, Z_WINDOW)
        adf = rolling_adf_pvalue(spread, ADF_WINDOW, ADF_STEP)

        z_now = float(z[-1]); p_now = float(adf[-1]); beta_now = float(beta[-1])
        spread_now = float(spread[-1])
        py_now = float(py_close[-1]); px_now = float(px_close[-1])

        telemetry = dict(
            z=z_now, adf_p=p_now, beta=beta_now, spread=spread_now,
            py=py_now, px=px_now, alpha=float(alpha[-1]),
        )

        if any(np.isnan(v) for v in (z_now, p_now, beta_now)):
            return Decision(action=Action.HOLD, telemetry=telemetry, reason="nan in signal")

        # ── Exit logic ──────────────────────────────────────────────────
        if position is not None:
            d = position["direction"]
            kill = p_now > self.p.adf_kill_max
            stopped = abs(z_now) > self.p.stop_z
            sig_exit = (d == +1 and z_now > -self.p.exit_z) or (d == -1 and z_now < self.p.exit_z)

            if kill or stopped or sig_exit:
                reason = "kill_switch" if kill else "stop" if stopped else "signal"
                close_y_buy = (d == -1)
                close_x_buy = (d == +1)
                py_close_px = y_data["top_ask"] if close_y_buy else y_data["top_bid"]
                px_close_px = x_data["top_ask"] if close_x_buy else x_data["top_bid"]
                return Decision(
                    action=Action.EXIT,
                    legs=[
                        Leg(self.p.pair_y, close_y_buy, position["sz_y"], py_close_px, reduce_only=True),
                        Leg(self.p.pair_x, close_x_buy, position["sz_x"], px_close_px, reduce_only=True),
                    ],
                    reason=reason,
                    telemetry=telemetry,
                )
            return Decision(action=Action.HOLD, telemetry=telemetry, reason="in position")

        # ── Entry logic ─────────────────────────────────────────────────
        cointegrated = p_now < self.p.adf_entry_max
        direction = None
        if cointegrated:
            if z_now < -self.p.entry_z:
                direction = +1   # long spread
            elif z_now > self.p.entry_z:
                direction = -1   # short spread
        if direction is None:
            return Decision(action=Action.HOLD, telemetry=telemetry,
                            reason="no entry signal" if cointegrated else "not cointegrated")

        sz_y, sz_x = leg_sizes(py_now, px_now, beta_now,
                               self.p.capital_usd, self.p.leverage)
        sz_y = round(sz_y, 4); sz_x = round(sz_x, 5)

        is_buy_y = (direction == +1)
        is_buy_x = (direction == -1)
        py_entry = y_data["top_bid"] if is_buy_y else y_data["top_ask"]
        px_entry = x_data["top_bid"] if is_buy_x else x_data["top_ask"]

        return Decision(
            action=Action.ENTER,
            legs=[
                Leg(self.p.pair_y, is_buy_y, sz_y, py_entry, reduce_only=False),
                Leg(self.p.pair_x, is_buy_x, sz_x, px_entry, reduce_only=False),
            ],
            telemetry=telemetry,
            state={
                "direction": direction,
                "sz_y": sz_y, "sz_x": sz_x,
                "entry_spread": spread_now,
                "entry_z": z_now,
            },
            safety_stop_pct=self.p.safety_stop_pct,
        )
