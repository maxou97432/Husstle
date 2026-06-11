import numpy as np


def rolling_ols_hedge(y: np.ndarray, x: np.ndarray, window: int = 240) -> np.ndarray:
    """
    Causal rolling OLS beta at each index i using observations [i-window, i).
    Uses prefix-sum trick: O(n) via cumulative sums of x, y, x², xy.
    Returns array of betas (NaN for first `window` bars).
    """
    n = len(y)
    beta = np.full(n, np.nan)

    cum_x = np.zeros(n + 1)
    cum_y = np.zeros(n + 1)
    cum_xx = np.zeros(n + 1)
    cum_xy = np.zeros(n + 1)

    for i in range(n):
        cum_x[i + 1] = cum_x[i] + x[i]
        cum_y[i + 1] = cum_y[i] + y[i]
        cum_xx[i + 1] = cum_xx[i] + x[i] * x[i]
        cum_xy[i + 1] = cum_xy[i] + x[i] * y[i]

    for i in range(window, n):
        lo = i - window
        sx = cum_x[i] - cum_x[lo]
        sy = cum_y[i] - cum_y[lo]
        sxx = cum_xx[i] - cum_xx[lo]
        sxy = cum_xy[i] - cum_xy[lo]
        denom = window * sxx - sx * sx
        if denom == 0:
            continue
        beta[i] = (window * sxy - sx * sy) / denom

    return beta
