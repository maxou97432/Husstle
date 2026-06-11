import numpy as np
from signal.hedge import rolling_ols_hedge


def compute_spread(
    y: np.ndarray,
    x: np.ndarray,
    hedge_window: int = 240,
) -> np.ndarray:
    """
    spread[i] = y[i] - beta[i] * x[i]
    beta computed causally over [i-hedge_window, i).
    """
    beta = rolling_ols_hedge(y, x, window=hedge_window)
    spread = y - beta * x
    return spread


def rolling_zscore(
    spread: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Causal z-score at index i: (spread[i] - mean([i-window,i))) / std([i-window,i))
    Uses prefix-sum trick for O(n).
    """
    n = len(spread)
    z = np.full(n, np.nan)

    cum_s = np.zeros(n + 1)
    cum_s2 = np.zeros(n + 1)

    for i in range(n):
        v = spread[i] if not np.isnan(spread[i]) else 0.0
        cum_s[i + 1] = cum_s[i] + v
        cum_s2[i + 1] = cum_s2[i] + v * v

    for i in range(window, n):
        if np.isnan(spread[i]):
            continue
        lo = i - window
        s = cum_s[i] - cum_s[lo]
        s2 = cum_s2[i] - cum_s2[lo]
        mean = s / window
        var = s2 / window - mean * mean
        if var <= 0:
            continue
        z[i] = (spread[i] - mean) / np.sqrt(var)

    return z
