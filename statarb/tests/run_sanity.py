"""
Sanity-test orchestrator. Runs, IN ORDER:
  1. Causality / anti-lookahead   (tests.test_causality)
  2. Shuffle test                 (permute returns -> edge must collapse to ~0)
  3. Random-entry benchmark       (random entries -> net PnL must be <= ~0)

NO strategy performance metric is reported until all three PASS.

Usage:
    python -m tests.run_sanity            # real data if present, else synthetic
    python -m tests.run_sanity --synthetic
    python -m tests.run_sanity --n 200    # permutations / sims
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal.spread import compute_spread, rolling_zscore
from signal.cointegration import rolling_adf_pvalue
from backtest.engine import BacktestConfig, run_backtest
from backtest.metrics import sharpe
from tests._data import get_data
from tests import test_causality
from config import HEDGE_WINDOW as HEDGE_W, Z_WINDOW as Z_W, ADF_WINDOW as ADF_W, ADF_STEP, BARS_PER_YEAR, INTERVAL


def build_signals(close_eth, close_btc):
    log_eth = np.log(close_eth)
    log_btc = np.log(close_btc)
    spread, beta, _ = compute_spread(log_eth, log_btc, HEDGE_W)
    z = rolling_zscore(spread, Z_W)
    adf = rolling_adf_pvalue(spread, ADF_W, ADF_STEP)
    return spread, z, adf, beta


def edge_of(close_eth, close_btc, spread, z, adf, fund_eth, fund_btc, beta, cfg, forced=None):
    trades, equity = run_backtest(
        close_eth, close_btc, spread, z, adf, fund_eth, fund_btc, beta, cfg, forced_entry=forced
    )
    return sharpe(equity, bars_per_year=BARS_PER_YEAR), float(equity[-1]), len(trades)


def _pct(arr, q):
    return float(np.percentile(arr, q))


# ────────────────────────────────────────────────────────────────────────────
# TEST 2 — SHUFFLE
# ────────────────────────────────────────────────────────────────────────────
def shuffle_test(data, cfg, n_perm=100, seed=42):
    close_eth, close_btc, fund_eth, fund_btc = data
    spread, z, adf, beta = build_signals(close_eth, close_btc)
    real_sharpe, real_pnl, real_n = edge_of(
        close_eth, close_btc, spread, z, adf, fund_eth, fund_btc, beta, cfg
    )

    log_eth = np.log(close_eth)
    log_btc = np.log(close_btc)
    r_eth = np.diff(log_eth)
    r_btc = np.diff(log_btc)
    e0, b0 = log_eth[0], log_btc[0]

    rng = np.random.default_rng(seed)
    null_sharpe, null_pnl = [], []
    for _ in range(n_perm):
        # JOINT permutation of return pairs: preserves the contemporaneous
        # ETH/BTC hedge relationship, destroys serial mean-reversion structure.
        perm = rng.permutation(len(r_eth))
        le = np.concatenate([[e0], e0 + np.cumsum(r_eth[perm])])
        lb = np.concatenate([[b0], b0 + np.cumsum(r_btc[perm])])
        sp, zz, ad, be = build_signals(np.exp(le), np.exp(lb))
        s, p, _ = edge_of(np.exp(le), np.exp(lb), sp, zz, ad, fund_eth, fund_btc, be, cfg)
        null_sharpe.append(s)
        null_pnl.append(p)

    null_sharpe = np.array(null_sharpe)
    null_pnl = np.array(null_pnl)
    ns_mean, ns_std = null_sharpe.mean(), null_sharpe.std()
    z_score = (real_sharpe - ns_mean) / ns_std if ns_std > 0 else float("inf")
    pctile = float((null_sharpe < real_sharpe).mean() * 100)

    print("=" * 60)
    print("TEST 2 — SHUFFLE TEST  ({} permutations)".format(n_perm))
    print("=" * 60)
    print(f"  Real strategy     : Sharpe={real_sharpe:+.3f}  netPnL={real_pnl:+.4f}  trades={real_n}")
    print(f"  Null Sharpe       : mean={ns_mean:+.3f}  std={ns_std:.3f}")
    print(f"     percentiles     p5={_pct(null_sharpe,5):+.3f}  p50={_pct(null_sharpe,50):+.3f}"
          f"  p95={_pct(null_sharpe,95):+.3f}  p99={_pct(null_sharpe,99):+.3f}")
    print(f"  Null netPnL       : mean={null_pnl.mean():+.4f}  p95={_pct(null_pnl,95):+.4f}")
    print(f"  Real vs null      : percentile={pctile:.1f}%   z-score={z_score:+.2f}σ")

    # ── Two SEPARATE conclusions ──
    # (a) MACHINERY validity (what this sanity test is for): a lookahead leak
    #     would survive shuffling and push the null POSITIVE. So machinery is
    #     sound iff the null is NOT positively biased (mean <= ~0; cost drag
    #     makes a clean null land slightly negative).
    se_null = ns_std / np.sqrt(n_perm) if n_perm > 0 else float("inf")
    null_not_positive = ns_mean < 2 * se_null            # not significantly > 0
    machinery_ok = null_not_positive and null_pnl.mean() <= 1e-6 + null_pnl.std()

    # (b) EDGE existence (informational, NOT a machinery verdict): is the real
    #     strategy a clear outlier vs the null?
    edge_detected = real_sharpe > _pct(null_sharpe, 95)

    print(f"\n  Machinery check   : null not positively biased "
          f"(mean={ns_mean:+.3f}, 2·SE={2*se_null:.3f}) -> "
          f"{'OK, no lookahead leak' if null_not_positive else 'LEAK SUSPECTED'}")
    print(f"  Edge check        : real {'IS' if edge_detected else 'is NOT'} "
          f"a clear outlier vs null p95")
    if not edge_detected:
        print(f"     ↳ diagnosis: machinery clean -> this is 'NO EDGE / insufficient "
              f"power' ({real_n} trades), NOT a backtest bug")

    verdict = machinery_ok
    print(f"\n  TEST 2 VERDICT (machinery): {'PASS' if verdict else 'FAIL'}")
    if not verdict:
        print(f"     ↳ null mean Sharpe {ns_mean:+.3f} is significantly POSITIVE "
              f"-> backtest manufactures edge from noise -> LOOKAHEAD BUG")
    return verdict, {"real_sharpe": real_sharpe, "null_mean": ns_mean,
                     "z": z_score, "pctile": pctile, "edge_detected": edge_detected}


# ────────────────────────────────────────────────────────────────────────────
# TEST 3 — RANDOM-ENTRY BENCHMARK
# ────────────────────────────────────────────────────────────────────────────
def random_entry_test(data, cfg, n_sims=100, seed=0):
    close_eth, close_btc, fund_eth, fund_btc = data
    spread, z, adf, beta = build_signals(close_eth, close_btc)

    # Real entry frequency among eligible (valid-signal) bars.
    eligible = np.where(~np.isnan(z) & ~np.isnan(beta) & ~np.isnan(spread))[0]
    eligible = eligible[eligible < len(z) - 1]
    _, _, real_n = edge_of(close_eth, close_btc, spread, z, adf, fund_eth, fund_btc, beta, cfg)
    freq = real_n / max(len(eligible), 1)

    rng = np.random.default_rng(seed)
    pnls, sharpes, counts = [], [], []
    for _ in range(n_sims):
        forced = np.full(len(z), np.nan)
        mask = rng.random(len(eligible)) < freq
        idx = eligible[mask]
        forced[idx] = rng.choice([-1.0, 1.0], size=len(idx))
        # Exits/sizing/funding/fees use the REAL z/adf machinery; only entries random.
        s, p, ntr = edge_of(close_eth, close_btc, spread, z, adf,
                            fund_eth, fund_btc, beta, cfg, forced=forced)
        pnls.append(p)
        sharpes.append(s)
        counts.append(ntr)

    pnls = np.array(pnls)
    sharpes = np.array(sharpes)
    frac_pos = float((pnls > 0).mean())
    mean_pnl = pnls.mean()
    se = pnls.std() / np.sqrt(len(pnls))
    t_stat = mean_pnl / se if se > 0 else 0.0

    print("=" * 60)
    print("TEST 3 — RANDOM-ENTRY BENCHMARK  ({} sims)".format(n_sims))
    print("=" * 60)
    print(f"  Real entry freq   : {freq:.4f}  (~{np.mean(counts):.0f} random trades/sim)")
    print(f"  Net PnL (fees+fund): mean={mean_pnl:+.5f}  std={pnls.std():.5f}  SE={se:.5f}")
    print(f"     percentiles     p5={_pct(pnls,5):+.5f}  p50={_pct(pnls,50):+.5f}  p95={_pct(pnls,95):+.5f}")
    print(f"  Net Sharpe        : mean={sharpes.mean():+.3f}  p95={_pct(sharpes,95):+.3f}")
    print(f"  Fraction PnL > 0  : {frac_pos:.1%}        t-stat(mean vs 0)={t_stat:+.2f}")

    # PASS: net PnL not significantly POSITIVE. Random entries pay fees+funding,
    # so the honest expectation is <= 0. A significant positive => fill/funding bias.
    verdict = t_stat < 2.0
    print(f"\n  TEST 3 VERDICT: {'PASS' if verdict else 'FAIL'}")
    if not verdict:
        print(f"     ↳ random-entry net PnL is significantly POSITIVE (t={t_stat:+.2f}) "
              f"-> fill/funding model biased in our favour")
    return verdict, {"mean_pnl": float(mean_pnl), "t_stat": float(t_stat), "frac_pos": frac_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    ap.add_argument("--n", type=int, default=100, help="permutations / sims")
    args = ap.parse_args()

    data, source = get_data(prefer_real=not args.synthetic)
    cfg = BacktestConfig()

    print(f"\nTimeframe: {INTERVAL}  (windows: hedge={HEDGE_W} z={Z_W} adf={ADF_W} step={ADF_STEP} bars/yr={BARS_PER_YEAR})")
    print(f"Data source: {source}\n")

    t1 = test_causality.run()
    print()
    if not t1:
        print(">>> Causality FAILED. Halting — no further tests, no performance shown.")
        sys.exit(1)

    t2, m2 = shuffle_test(data, cfg, n_perm=args.n)
    print()
    if not t2:
        print(">>> Shuffle MACHINERY FAILED (lookahead). Halting — no performance shown.")
        sys.exit(1)

    t3, _ = random_entry_test(data, cfg, n_sims=args.n)
    print()
    if not t3:
        print(">>> Random-entry FAILED (fill/funding bias). Halting — no performance shown.")
        sys.exit(1)

    print("=" * 60)
    print("  ALL 3 SANITY TESTS PASSED — backtest machinery is trustworthy:")
    print("    1. no lookahead (causality)")
    print("    2. no manufactured edge from noise (shuffle null collapses)")
    print("    3. no fill/funding bias (random entry ~0)")
    print("-" * 60)
    if m2.get("edge_detected"):
        print("  Edge: real strategy IS a clear outlier vs the null.")
        print("  -> You may now compute full performance metrics (run_backtest.py).")
    else:
        print("  Edge: NONE detected at placeholder thresholds 2.0/0.5/3.5.")
        print(f"        Real Sharpe={m2['real_sharpe']:+.3f}, percentile vs null={m2['pctile']:.0f}%.")
        print("        The machinery is sound, but there is no edge to report yet —")
        print("        too few trades for statistical power (brief §6). Next step is")
        print("        threshold bootstrapping / lower timeframe, NOT a GO verdict.")
    print("=" * 60)


if __name__ == "__main__":
    main()
