"""
Test 1 — Causality / anti-lookahead.

For every signal function, the value produced at bar t must depend ONLY on
data up to (and including) bar t, never on data strictly after t.

Method: compute value_t on the original series. Then arbitrarily corrupt all
data at indices > t and recompute. The value at t (and everything before)
must be byte-for-byte identical. If it changes, the function peeks into the
future -> FAIL, and we name the culprit.

Runnable both as a script (python -m tests.test_causality) and via pytest.
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal.hedge import rolling_ols_hedge
from signal.spread import compute_spread, rolling_zscore
from signal.cointegration import rolling_adf_pvalue

HEDGE_W = 240
Z_W = 60
ADF_W = 240
ADF_STEP = 6

TEST_BARS = [250, 400, 600, 900]   # bars at which to check causality


def _corrupt_after(arr: np.ndarray, t: int, rng) -> np.ndarray:
    """Return a copy of arr with indices > t replaced by arbitrary noise."""
    out = arr.copy()
    if t + 1 < len(out):
        out[t + 1:] = rng.normal(out.mean(), out.std() * 5 + 1.0, len(out) - t - 1)
    return out


def _equal_prefix(a: np.ndarray, b: np.ndarray, t: int) -> tuple[bool, int]:
    """True if a[:t+1] == b[:t+1] (NaN==NaN). Returns (ok, first_diff_index)."""
    pa, pb = a[: t + 1], b[: t + 1]
    both_nan = np.isnan(pa) & np.isnan(pb)
    eq = (pa == pb) | both_nan
    if eq.all():
        return True, -1
    return False, int(np.argmin(eq))


def _make_series(n: int = 1200, seed: int = 3):
    rng = np.random.default_rng(seed)
    log_btc = np.cumsum(rng.normal(0, 0.015, n)) + np.log(40000.0)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - 0.05 * s[t - 1] + rng.normal(0, 0.02)
    log_eth = 0.5 + 1.0 * log_btc + s
    return log_eth, log_btc


def check_ols() -> list[str]:
    fails = []
    rng = np.random.default_rng(11)
    y, x = _make_series()
    beta, alpha = rolling_ols_hedge(y, x, HEDGE_W)
    for t in TEST_BARS:
        y2 = _corrupt_after(y, t, rng)
        x2 = _corrupt_after(x, t, rng)
        b2, a2 = rolling_ols_hedge(y2, x2, HEDGE_W)
        ok_b, idx_b = _equal_prefix(beta, b2, t)
        ok_a, idx_a = _equal_prefix(alpha, a2, t)
        if not ok_b:
            fails.append(f"rolling_ols_hedge β leaks future: diff at idx {idx_b} (t={t})")
        if not ok_a:
            fails.append(f"rolling_ols_hedge α leaks future: diff at idx {idx_a} (t={t})")
    return fails


def check_spread() -> list[str]:
    fails = []
    rng = np.random.default_rng(12)
    y, x = _make_series()
    spread, _, _ = compute_spread(y, x, HEDGE_W)
    for t in TEST_BARS:
        y2 = _corrupt_after(y, t, rng)
        x2 = _corrupt_after(x, t, rng)
        s2, _, _ = compute_spread(y2, x2, HEDGE_W)
        ok, idx = _equal_prefix(spread, s2, t)
        if not ok:
            fails.append(f"compute_spread leaks future: diff at idx {idx} (t={t})")
    return fails


def check_zscore() -> list[str]:
    fails = []
    rng = np.random.default_rng(13)
    y, x = _make_series()
    spread, _, _ = compute_spread(y, x, HEDGE_W)
    z = rolling_zscore(spread, Z_W)
    for t in TEST_BARS:
        sp2 = _corrupt_after(spread, t, rng)
        z2 = rolling_zscore(sp2, Z_W)
        ok, idx = _equal_prefix(z, z2, t)
        if not ok:
            fails.append(f"rolling_zscore leaks future: diff at idx {idx} (t={t})")
    return fails


def check_adf() -> list[str]:
    fails = []
    rng = np.random.default_rng(14)
    y, x = _make_series()
    spread, _, _ = compute_spread(y, x, HEDGE_W)
    pval = rolling_adf_pvalue(spread, ADF_W, ADF_STEP)
    for t in TEST_BARS:
        sp2 = _corrupt_after(spread, t, rng)
        p2 = rolling_adf_pvalue(sp2, ADF_W, ADF_STEP)
        ok, idx = _equal_prefix(pval, p2, t)
        if not ok:
            fails.append(f"rolling_adf_pvalue leaks future: diff at idx {idx} (t={t})")
    return fails


def check_pipeline_end_to_end() -> list[str]:
    """Perturb raw prices > t; β, spread, z, ADF prefixes must all be unchanged."""
    fails = []
    rng = np.random.default_rng(15)
    y, x = _make_series()
    beta, alpha = rolling_ols_hedge(y, x, HEDGE_W)
    spread, _, _ = compute_spread(y, x, HEDGE_W)
    z = rolling_zscore(spread, Z_W)
    pval = rolling_adf_pvalue(spread, ADF_W, ADF_STEP)
    for t in TEST_BARS:
        y2 = _corrupt_after(y, t, rng)
        x2 = _corrupt_after(x, t, rng)
        b2, _ = rolling_ols_hedge(y2, x2, HEDGE_W)
        s2, _, _ = compute_spread(y2, x2, HEDGE_W)
        z2 = rolling_zscore(s2, Z_W)
        p2 = rolling_adf_pvalue(s2, ADF_W, ADF_STEP)
        for name, a, b in [("β", beta, b2), ("spread", spread, s2), ("z", z, z2), ("ADF", pval, p2)]:
            ok, idx = _equal_prefix(a, b, t)
            if not ok:
                fails.append(f"end-to-end {name} leaks future: diff at idx {idx} (t={t})")
    return fails


CHECKS = {
    "OLS β/α": check_ols,
    "spread": check_spread,
    "z-score": check_zscore,
    "ADF p-value": check_adf,
    "end-to-end pipeline": check_pipeline_end_to_end,
}


def run() -> bool:
    print("=" * 60)
    print("TEST 1 — CAUSALITY / ANTI-LOOKAHEAD")
    print("=" * 60)
    all_fails = []
    for name, fn in CHECKS.items():
        fails = fn()
        status = "PASS" if not fails else "FAIL"
        print(f"  [{status}] {name}")
        for f in fails:
            print(f"         ↳ {f}")
        all_fails.extend(fails)
    verdict = not all_fails
    print(f"\n  TEST 1 VERDICT: {'PASS' if verdict else 'FAIL'} "
          f"({len(all_fails)} lookahead violation(s))")
    return verdict


# pytest entry points
def test_ols_causal():        assert not check_ols()
def test_spread_causal():     assert not check_spread()
def test_zscore_causal():     assert not check_zscore()
def test_adf_causal():        assert not check_adf()
def test_pipeline_causal():   assert not check_pipeline_end_to_end()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
