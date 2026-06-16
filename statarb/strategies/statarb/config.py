"""
StatArb-specific configuration: universe, signal windows, default thresholds.
"""
from __future__ import annotations

from core.config import BARS_PER_DAY

# Asset universe traded by the StatArb strategy.
UNIVERSE = ["BTC", "ETH", "SOL", "ARB", "AVAX", "BNB"]

# Calendar-anchored windows (auto-rescale when INTERVAL changes).
HEDGE_WINDOW = 40 * BARS_PER_DAY                  # ~40 days OLS lookback
Z_WINDOW     = 10 * BARS_PER_DAY                  # ~10 days z-score
ADF_WINDOW   = 40 * BARS_PER_DAY                  # ~40 days cointegration test
ADF_STEP     = 1  * BARS_PER_DAY                  # recompute ADF daily

# Default signal thresholds (placeholders, calibrated via tools/sweep_thresholds.py)
ENTRY_Z = 2.0
EXIT_Z  = 0.5
STOP_Z  = 3.5
ADF_ENTRY_MAX = 0.05
ADF_KILL_MAX  = 0.10
