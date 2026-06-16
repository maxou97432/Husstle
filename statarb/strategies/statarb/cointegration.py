import numpy as np
from statsmodels.tsa.stattools import adfuller


def rolling_adf_pvalue(
    spread: np.ndarray,
    window: int = 240,
    step: int = 6,
) -> np.ndarray:
    """
    Causal rolling ADF p-value computed every `step` bars, carry-forward between steps.
    Returns array of p-values (NaN until first window is available).
    """
    n = len(spread)
    pval = np.full(n, np.nan)
    last_p = np.nan

    for i in range(window, n):
        if (i - window) % step != 0:
            pval[i] = last_p
            continue

        window_data = spread[i - window: i]
        if np.any(np.isnan(window_data)):
            pval[i] = last_p
            continue

        try:
            result = adfuller(window_data, autolag="AIC")
            last_p = result[1]
        except Exception:
            pass

        pval[i] = last_p

    return pval
