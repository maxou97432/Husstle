"""
Sizing & risk parameters per the brief §2.

- Leverage: 2× per leg (~4× gross).
- Margin mode: isolated.
- Equal-dollar sizing on the two legs via β (so the spread trade is
  market-neutral by construction).
- Position size derives from a `capital_per_trade` budget and current prices.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    leverage_per_leg: int = 2
    capital_per_trade_usd: float = 100.0    # testnet default; bump for live
    min_notional_usd: float = 10.0
    margin_mode: str = "isolated"


def leg_sizes(
    px_y: float, px_x: float, beta: float,
    capital_usd: float, leverage: int,
) -> tuple[float, float]:
    """
    Compute size_y, size_x (in coin units) such that:
      - dollar notional of y-leg : dollar notional of x-leg = 1 : |beta|
      - y-leg notional = capital_usd * leverage / (1 + |beta|)
    Returned sizes are always POSITIVE (magnitudes). Direction is applied
    by the caller (buy/sell).
    """
    total_notional = capital_usd * leverage
    abs_b = abs(beta) if beta else 1.0
    notional_y = total_notional / (1.0 + abs_b)
    notional_x = total_notional - notional_y
    sz_y = notional_y / px_y
    sz_x = notional_x / px_x
    return sz_y, sz_x


def check_min_notional(px: float, sz: float, min_notional: float) -> bool:
    return px * sz >= min_notional
