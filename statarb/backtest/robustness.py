from __future__ import annotations

import copy
import random
import numpy as np
from backtest.engine import BacktestConfig, run_backtest, Trade
from backtest.metrics import evaluate, sharpe


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
        adf_pval[cut:], funding_y[cut:] if funding_y is not None else None,
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
    Vary entry_z and exit_z by ±20%, 16 combinations. Returns list of result dicts.
    """
    results = []
    entry_vals = [base_cfg.entry_z * (1 - delta), base_cfg.entry_z, base_cfg.entry_z * (1 + delta)]
    exit_vals = [base_cfg.exit_z, base_cfg.exit_z + delta]
    adf_vals = [base_cfg.adf_pval_max * (1 - delta), base_cfg.adf_pval_max, base_cfg.adf_pval_max * (1 + delta)]

    combos: list[tuple] = []
    for ez in entry_vals:
        for exz in exit_vals:
            for ap in adf_vals:
                combos.append((ez, exz, ap))
    combos = combos[:n_boot]

    for (ez, exz, ap) in combos:
        cfg = copy.copy(base_cfg)
        cfg.entry_z = ez
        cfg.exit_z = exz
        cfg.adf_pval_max = ap
        trades, equity = run_backtest(
            close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta, cfg
        )
        res = evaluate(trades, equity)
        res["label"] = f"thresh_ez={ez:.2f}_exz={exz:.2f}_adf={ap:.3f}"
        results.append(res)

    return results


def shuffle_test(
    base_cfg: BacktestConfig,
    close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta,
    n_shuffles: int = 100,
    seed: int = 42,
) -> dict:
    """
    Shuffle the zscore series and rerun. Baseline null distribution.
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
        "null_sharpe_mean": float(np.mean(sharpes)),
        "null_sharpe_p95": float(np.percentile(sharpes, 95)),
    }


def random_entry_bench(
    base_cfg: BacktestConfig,
    close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta,
    n_sims: int = 100,
    seed: int = 0,
) -> dict:
    """
    Replace entry signal with random ±1 entries at the same frequency as the real strategy.
    """
    rng = np.random.default_rng(seed)
    n = len(zscore)
    valid = np.where(~np.isnan(zscore))[0]
    sharpes = []

    real_trades, _ = run_backtest(
        close_y, close_x, spread, zscore, adf_pval, funding_y, funding_x, hedge_beta, base_cfg
    )
    freq = len(real_trades) / max(len(valid), 1)

    for _ in range(n_sims):
        z_rand = np.full(n, np.nan)
        entry_mask = rng.random(len(valid)) < freq
        for idx in valid[entry_mask]:
            z_rand[idx] = rng.choice([-base_cfg.entry_z - 0.01, base_cfg.entry_z + 0.01])
        trades, equity = run_backtest(
            close_y, close_x, spread, z_rand, adf_pval, funding_y, funding_x, hedge_beta, base_cfg
        )
        sharpes.append(sharpe(equity))

    return {
        "label": "random_entry_bench",
        "bench_sharpe_mean": float(np.mean(sharpes)),
        "bench_sharpe_p95": float(np.percentile(sharpes, 95)),
    }
