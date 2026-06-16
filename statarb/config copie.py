"""
Single source of truth for the candle timeframe and all window/annualisation
constants derived from it. Switching to a new TF only requires editing INTERVAL.

Brief calendar-anchored windows (kept invariant across TF changes):
  - hedge OLS  : ~40 days
  - z-score    : ~10 days
  - ADF        : ~40 days
  - ADF step   :  ~1 day
"""
from __future__ import annotations

INTERVAL = "1h"          # "1h" or "4h"
INTERVAL_HOURS = {"1h": 1, "4h": 4}[INTERVAL]
INTERVAL_MS = INTERVAL_HOURS * 3600 * 1000

BARS_PER_DAY = 24 // INTERVAL_HOURS
BARS_PER_YEAR = 365 * BARS_PER_DAY

# Calendar-anchored windows (recomputed when TF changes)
HEDGE_WINDOW = 40 * BARS_PER_DAY   # ~40d
Z_WINDOW     = 10 * BARS_PER_DAY   # ~10d
ADF_WINDOW   = 40 * BARS_PER_DAY   # ~40d
ADF_STEP     = 1  * BARS_PER_DAY   # ~1d

LOOKBACK_DAYS_DEFAULT = 730        # 2 years

# Multi-pair StatArb universe (probed liquid on HL with 1h history).
# 6 assets -> 15 unordered pairs. Mix L1 cores + L2 + memecoin for diversity.
UNIVERSE = ["BTC", "ETH", "SOL", "ARB", "AVAX", "BNB"]
