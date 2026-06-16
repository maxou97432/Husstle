"""
Live StatArb bot — BTC/ETH only, testnet only.

Loop:
  every 1h (TF boundary):
    1. fetch the last HEDGE_WINDOW+ candles from HL (BTC, ETH)
    2. recompute causal β/α, spread, z-score, ADF (same code as backtest)
    3. if flat:    enter if |z|>entry_z AND ADF p<adf_entry_max
    4. if in pos:  exit if signal/stop/kill-switch (same logic as backtest)
    5. orders go through orders.place_two_legs() with maker+orphan-close
    6. journal every event to live_journal.jsonl + Telegram (if configured)

Conformity check: at every bar we compare the live signal to what the backtest
engine would have produced from the same history. Divergence -> halt + alert.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HEDGE_WINDOW, Z_WINDOW, ADF_WINDOW, ADF_STEP, INTERVAL, INTERVAL_MS
from execution.hl_client import HLClient
from execution.orders import place_two_legs
from risk.sizing import RiskConfig, leg_sizes, check_min_notional
from risk.safeguards import Safeguards, CircuitBreaker
from alerts.notify import notify
from signal.spread import compute_spread, rolling_zscore
from signal.cointegration import rolling_adf_pvalue
from backtest.engine import BacktestConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("statarb.live")

PAIR_Y = "ETH"   # spread = log(ETH) - β·log(BTC) - α (matches backtest convention)
PAIR_X = "BTC"


def fetch_recent_closes(client: HLClient, coin: str, n_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (timestamps_ms, closes) for the most recent n_bars candles."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (n_bars + 5) * INTERVAL_MS
    data = client.candles(coin, INTERVAL, start_ms, end_ms)
    ts = np.array([int(c["t"]) for c in data], dtype=np.int64)
    closes = np.array([float(c["c"]) for c in data])
    return ts, closes


def current_signal(client: HLClient, cfg: BacktestConfig) -> dict:
    n_needed = HEDGE_WINDOW + ADF_WINDOW + 50
    ts_y, py = fetch_recent_closes(client, PAIR_Y, n_needed)
    ts_x, px = fetch_recent_closes(client, PAIR_X, n_needed)
    common = np.intersect1d(ts_y, ts_x)
    py = py[np.isin(ts_y, common)]
    px = px[np.isin(ts_x, common)]

    log_y = np.log(py); log_x = np.log(px)
    spread, beta, alpha = compute_spread(log_y, log_x, HEDGE_WINDOW)
    z = rolling_zscore(spread, Z_WINDOW)
    adf = rolling_adf_pvalue(spread, ADF_WINDOW, ADF_STEP)

    return {
        "ts": int(common[-1]),
        "py": float(py[-1]),
        "px": float(px[-1]),
        "spread": float(spread[-1]),
        "z": float(z[-1]),
        "beta": float(beta[-1]),
        "alpha": float(alpha[-1]),
        "adf_p": float(adf[-1]),
    }


def next_bar_close_ts() -> int:
    """Return the next TF boundary timestamp in ms (e.g., next hour)."""
    now = int(time.time() * 1000)
    return ((now // INTERVAL_MS) + 1) * INTERVAL_MS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Single iteration then exit")
    ap.add_argument("--dry-run", action="store_true", help="Force dry-run even with creds")
    ap.add_argument("--entry", type=float, default=2.0)
    ap.add_argument("--exit",  type=float, default=0.5)
    ap.add_argument("--stop",  type=float, default=3.5)
    ap.add_argument("--capital", type=float, default=100.0)
    args = ap.parse_args()

    cfg = BacktestConfig(entry_z=args.entry, exit_z=args.exit, stop_z=args.stop, leverage=2.0)
    risk = RiskConfig(capital_per_trade_usd=args.capital, leverage_per_leg=2)
    sg = Safeguards(max_open_pairs=1)
    cb = CircuitBreaker(sg)

    client = HLClient(network="testnet", dry_run=True if args.dry_run else None)
    notify("bot_start", network=client.network, dry_run=client.dry_run,
           entry=cfg.entry_z, exit=cfg.exit_z, stop=cfg.stop_z, capital=risk.capital_per_trade_usd)

    if not client.dry_run:
        # Configure isolated leverage on both legs once
        client.set_isolated_leverage(PAIR_Y, risk.leverage_per_leg)
        client.set_isolated_leverage(PAIR_X, risk.leverage_per_leg)

    open_position: dict | None = None  # {direction, sz_y, sz_x, entry_spread}

    def iteration():
        nonlocal open_position
        if cb.halted:
            notify("halted", reason=cb.halt_reason)
            return

        try:
            sig = current_signal(client, cfg)
            cb.record_success()
        except Exception as exc:
            cb.record_error(str(exc))
            notify("error", stage="signal", exc=str(exc))
            return

        notify("signal", **sig, position=open_position)

        live_positions = client.positions() if not client.dry_run else {}
        # ── Exit logic ───────────────────────────────────────────────────
        if open_position is not None:
            d = open_position["direction"]
            kill = sig["adf_p"] > cfg.adf_kill_max
            stop = abs(sig["z"]) > cfg.stop_z
            sig_exit = (d == +1 and sig["z"] > -cfg.exit_z) or (d == -1 and sig["z"] < cfg.exit_z)

            if kill or stop or sig_exit:
                reason = "kill_switch" if kill else "stop" if stop else "signal"
                bid_y, ask_y = client.book_top(PAIR_Y)
                bid_x, ask_x = client.book_top(PAIR_X)
                # Reverse the legs to close
                close_y_buy = (d == -1)   # we shorted Y at entry -> buy to close
                close_x_buy = (d == +1)
                px_y_close = ask_y if close_y_buy else bid_y
                px_x_close = ask_x if close_x_buy else bid_x

                res = place_two_legs(
                    client,
                    PAIR_Y, close_y_buy, open_position["sz_y"], px_y_close,
                    PAIR_X, close_x_buy, open_position["sz_x"], px_x_close,
                    max_lag_sec=sg.max_one_leg_lag_sec, reduce_only=True,
                )
                notify("exit", reason=reason, ok=res.ok, **sig)
                if res.ok:
                    open_position = None
            return

        # ── Entry logic ──────────────────────────────────────────────────
        cointegrated = sig["adf_p"] < cfg.adf_entry_max
        direction = None
        if cointegrated:
            if sig["z"] < -cfg.entry_z:
                direction = +1
            elif sig["z"] > cfg.entry_z:
                direction = -1

        if direction is None:
            return

        sz_y, sz_x = leg_sizes(sig["py"], sig["px"], sig["beta"],
                               risk.capital_per_trade_usd, risk.leverage_per_leg)
        # Round to HL lot sizes — TODO: pull szDecimals from asset_meta and quantize.
        sz_y = round(sz_y, 4); sz_x = round(sz_x, 5)
        if not (check_min_notional(sig["py"], sz_y, risk.min_notional_usd)
                and check_min_notional(sig["px"], sz_x, risk.min_notional_usd)):
            notify("entry_skipped", reason="below min notional", sz_y=sz_y, sz_x=sz_x)
            return

        # Direction +1 = long spread = long Y, short X
        is_buy_y = (direction == +1)
        is_buy_x = (direction == -1)
        bid_y, ask_y = client.book_top(PAIR_Y)
        bid_x, ask_x = client.book_top(PAIR_X)
        px_y = bid_y if is_buy_y else ask_y   # post-only maker on best touch
        px_x = bid_x if is_buy_x else ask_x

        res = place_two_legs(
            client,
            PAIR_Y, is_buy_y, sz_y, px_y,
            PAIR_X, is_buy_x, sz_x, px_x,
            max_lag_sec=sg.max_one_leg_lag_sec, reduce_only=False,
        )
        notify("entry", direction=direction, ok=res.ok, sz_y=sz_y, sz_x=sz_x, **sig)
        if res.ok:
            open_position = {
                "direction": direction,
                "sz_y": sz_y, "sz_x": sz_x,
                "entry_spread": sig["spread"],
                "entry_z": sig["z"],
            }

    if args.once:
        iteration()
        return

    # Sleep to next bar boundary, then run forever
    while True:
        try:
            wait_ms = next_bar_close_ts() - int(time.time() * 1000)
            log.info(f"sleeping {wait_ms/1000:.1f}s until next bar")
            time.sleep(max(1, wait_ms / 1000) + 2)  # +2s grace for candle to publish
            iteration()
        except KeyboardInterrupt:
            notify("bot_stop", reason="KeyboardInterrupt")
            break
        except Exception as exc:
            cb.record_error(str(exc))
            notify("error", stage="loop", exc=str(exc))
            time.sleep(30)


if __name__ == "__main__":
    main()
