"""
Shared configuration used by all strategies — exchange-side defaults.
Strategy-specific config lives under strategies/<name>/config.py.
"""
from __future__ import annotations

# Candle timeframe used across the codebase.
INTERVAL = "1h"                                   # "1h" or "4h"
INTERVAL_HOURS = {"1h": 1, "4h": 4}[INTERVAL]
INTERVAL_MS = INTERVAL_HOURS * 3600 * 1000

BARS_PER_DAY = 24 // INTERVAL_HOURS
BARS_PER_YEAR = 365 * BARS_PER_DAY

LOOKBACK_DAYS_DEFAULT = 730                       # 2 years of history requested
