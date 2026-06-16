from __future__ import annotations

import copy
import numpy as np
from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.metrics import evaluate, sharpe


def stress_fees(
    base_cfg: BacktestConfig,
    close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta,
) -> dict:
    cfg2 = copy.copy(base_cfg)
    cfg2.fee_stress = 2.0
    trades, equity = run_backtest(
        close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta, cfg2
    )
    return {"label": "stress_fees_2x", **evaluate(trades, equity)}


def out_of_sample(
    base_cfg: BacktestConfig,
    close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta,
    split: float = 0.70,
) -> dict:
    n = len(close_y)
    cut = int(n * split)
    trades, equity = run_backtest(
        close_y[cut:], close_x[cut:], spread[cut:], zscore[cut:],
        adf_pval[cut:],
        funding_y[cut:] if funding_y is not None else None,
        funding_x[cut:] if funding_x is not None else None,
        hedge_beta[cut:], base_cfg,
    )
    return {"label": "oos_30pct", **evaluate(trades, equity)}


def bootstrap_thresholds(
    base_cfg: BacktestConfig,
    close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta,
    n_boot: int = 16,
    delta: float = 0.20,
) -> list[dict]:
    """
    Perturb entry_z, exit_z, stop_z by ±delta (±20%). Gate: ≥75% of variants pass.
    """
    results = []
    combos: list[tuple] = []
    for ez_mult in (1 - delta, 1.0, 1 + delta):
        for exz_mult in (1 - delta, 1.0, 1 + delta):
            for sz_mult in (1 - delta, 1.0, 1 + delta):
                combos.append((ez_mult, exz_mult, sz_mult))
    combos = combos[:n_boot]

    for (ez_m, exz_m, sz_m) in combos:
        cfg = copy.copy(base_cfg)
        cfg.entry_z = base_cfg.entry_z * ez_m
        cfg.exit_z = base_cfg.exit_z * exz_m
        cfg.stop_z = base_cfg.stop_z * sz_m
        trades, equity = run_backtest(
            close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta, cfg
        )
        res = evaluate(trades, equity)
        res["label"] = f"thresh_ez={cfg.entry_z:.2f}_exz={cfg.exit_z:.2f}_sz={cfg.stop_z:.2f}"
        results.append(res)

    return results


def shuffle_test(
    base_cfg: BacktestConfig,
    close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta,
    n_shuffles: int = 100,
    seed: int = 42,
) -> dict:
    """
    Permute the zscore series randomly. A real edge must fall to ~0 under shuffling.
    """
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_shuffles):
        z_shuf = zscore.copy()
        valid = ~np.isnan(z_shuf)
        z_shuf[valid] = rng.permutation(z_shuf[valid])
        trades, equity = run_backtest(
            close_y, close_x, spread, z_shuf, adf_pval, funding_y, funding_x, hedge_beta, base_cfg
        )
        sharpes.append(sharpe(equity))

    return {
        "label": "shuffle_test",
        "null_sharpe_mean": round(float(np.mean(sharpes)), 3),
        "null_sharpe_p95": round(float(np.percentile(sharpes, 95)), 3),
    }


def random_entry_bench(
    base_cfg: BacktestConfig,
    close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta,
    n_sims: int = 100,
    seed: int = 0,
) -> dict:
    """
    Random entries at the same frequency as the real strategy. P&L should be ≈0 net of fees.
    """
    rng = np.random.default_rng(seed)
    n = len(zscore)
    valid = np.where(~np.isnan(zscore))[0]

    real_trades, _ = run_backtest(
        close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta, base_cfg
    )
    freq = len(real_trades) / max(len(valid), 1)

    sharpes = []
    for _ in range(n_sims):
        z_rand = np.full(n, np.nan)
        entry_mask = rng.random(len(valid)) < freq
        for idx in valid[entry_mask]:
            z_rand[idx] = rng.choice([
                -(base_cfg.entry_z + 0.01),
                base_cfg.entry_z + 0.01,
            ])
        trades, equity = run_backtest(
            close_y, close_x, spread, z_rand, adf_pval, funding_y, funding_x, hedge_beta, base_cfg
        )
        sharpes.append(sharpe(equity))

    return {
        "label": "random_entry_bench",
        "bench_sharpe_mean": round(float(np.mean(sharpes)), 3),
        "bench_sharpe_p95": round(float(np.percentile(sharpes, 95)), 3),
    }
