"""
Shared data loader for sanity tests.

Returns aligned numpy arrays (close_eth, close_btc, fund_eth, fund_btc).
Prefers REAL Hyperliquid data from DuckDB; falls back to SYNTHETIC
cointegrated data with a *known* mean-reverting edge so the sanity tests
have ground truth (real strategy should have edge; shuffled should not).
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _warn_fallback(reason: str) -> None:
    print(f"[_data] real-data load failed -> {reason}", file=sys.stderr)


def load_real() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from data.store import get_conn, load_candles_pl, load_funding_pl
        conn = get_conn()
        btc = load_candles_pl(conn, "BTC").sort("ts").collect()
        eth = load_candles_pl(conn, "ETH").sort("ts").collect()
        fb = load_funding_pl(conn, "BTC").sort("ts").collect()
        fe = load_funding_pl(conn, "ETH").sort("ts").collect()
        conn.close()
    except Exception as exc:
        _warn_fallback(f"{type(exc).__name__}: {exc}")
        return None

    if btc.height == 0 or eth.height == 0:
        _warn_fallback("empty candles table (run fetch first)")
        return None

    btc_ts = set(btc["ts"].to_list())
    eth_ts = set(eth["ts"].to_list())
    common = sorted(btc_ts & eth_ts)
    if len(common) < 400:
        _warn_fallback(f"only {len(common)} common bars (<400)")
        return None

    common_set = set(common)
    btc_map = dict(zip(btc["ts"].to_list(), btc["close"].to_list()))
    eth_map = dict(zip(eth["ts"].to_list(), eth["close"].to_list()))
    fb_map = dict(zip(fb["ts"].to_list(), fb["rate"].to_list())) if fb.height else {}
    fe_map = dict(zip(fe["ts"].to_list(), fe["rate"].to_list())) if fe.height else {}

    ts = np.array(common, dtype=np.int64)
    close_btc = np.array([btc_map[t] for t in common])
    close_eth = np.array([eth_map[t] for t in common])

    def agg(fmap):
        if not fmap:
            return np.zeros(len(ts))
        f_ts = np.array(sorted(fmap), dtype=np.int64)
        f_rate = np.array([fmap[t] for t in sorted(fmap)])
        out = np.zeros(len(ts))
        iv = 4 * 3600 * 1000
        for i, t in enumerate(ts):
            m = (f_ts > t - iv) & (f_ts <= t)
            out[i] = f_rate[m].sum() if m.any() else 0.0
        return out

    return close_eth, close_btc, agg(fe_map), agg(fb_map)


def make_synthetic(
    n: int = 3000,
    beta_true: float = 1.0,
    alpha_true: float = 0.5,
    kappa: float = 0.05,      # OU mean-reversion speed (per 4h bar)
    sigma_spread: float = 0.02,
    sigma_btc: float = 0.015,
    funding_sigma: float = 1e-4,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    log_btc : random walk.
    log_eth = alpha_true + beta_true*log_btc + s_t, where s_t is a mean-reverting
    Ornstein-Uhlenbeck process -> ETH/BTC are cointegrated with a tradeable,
    mean-reverting spread (the real strategy SHOULD have an edge here).
    """
    rng = np.random.default_rng(seed)

    log_btc = np.cumsum(rng.normal(0, sigma_btc, n)) + np.log(40000.0)

    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - kappa * s[t - 1] + rng.normal(0, sigma_spread)

    log_eth = alpha_true + beta_true * log_btc + s

    close_btc = np.exp(log_btc)
    close_eth = np.exp(log_eth)

    fund_btc = rng.normal(0, funding_sigma, n)
    fund_eth = rng.normal(0, funding_sigma, n)
    return close_eth, close_btc, fund_eth, fund_btc


def get_data(prefer_real: bool = True) -> tuple[tuple, str]:
    """Return ((close_eth, close_btc, fund_eth, fund_btc), source_label)."""
    if prefer_real:
        real = load_real()
        if real is not None:
            return real, "REAL (Hyperliquid/DuckDB)"
    return make_synthetic(), "SYNTHETIC (cointegrated OU spread)"
