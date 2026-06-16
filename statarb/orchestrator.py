"""
Multi-strategy orchestrator.

Drives N Strategy instances each TF bar:
  1. fetch market data once per coin (deduplicated across strategies)
  2. for each strategy → compute_signal() → translate to HL orders
  3. journal everything, apply safeguards

Strategies declare which coins they need; the orchestrator handles I/O,
order placement, partial fills, safety stops, and circuit breakers.

This replaces live_bot.py's monolithic loop. Each Strategy is a black box.
"""
from __future__ import annotations

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import INTERVAL, INTERVAL_MS
from core.execution.hl_client import HLClient
from core.execution.orders import place_two_legs
from core.risk.safeguards import Safeguards, CircuitBreaker
from core.alerts.notify import notify
from strategies.base import Strategy, Decision, Action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("statarb.orchestrator")


def _fetch_close_window(client: HLClient, coin: str, n_bars: int) -> tuple[np.ndarray, np.ndarray]:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (n_bars + 5) * INTERVAL_MS
    data = client.candles(coin, INTERVAL, start_ms, end_ms)
    ts = np.array([int(c["t"]) for c in data], dtype=np.int64)
    closes = np.array([float(c["c"]) for c in data])
    return ts, closes


def _build_market_data(client: HLClient, coins: list[str], n_bars: int) -> dict:
    """Fetch each unique coin once; return dict[coin] = {ts, close, top_bid, top_ask}."""
    out = {}
    for coin in set(coins):
        ts, closes = _fetch_close_window(client, coin, n_bars)
        bid, ask = client.book_top(coin)
        out[coin] = {"ts": ts, "close": closes, "top_bid": bid, "top_ask": ask}
    return out


def next_bar_close_ts() -> int:
    now = int(time.time() * 1000)
    return ((now // INTERVAL_MS) + 1) * INTERVAL_MS


class Orchestrator:
    def __init__(self, strategies: list[Strategy], client: HLClient, safeguards: Safeguards):
        self.strategies = strategies
        self.client = client
        self.cb = CircuitBreaker(safeguards)
        self.sg = safeguards
        self.positions: dict[str, dict | None] = {s.name: None for s in strategies}

    def _execute_decision(self, strat: Strategy, decision: Decision) -> None:
        sname = strat.name

        if decision.action == Action.HOLD:
            notify("hold", strategy=sname, reason=decision.reason, **decision.telemetry,
                   position=self.positions[sname])
            return

        if decision.action == Action.ENTER:
            if len(decision.legs) != 2:
                notify("error", strategy=sname, stage="enter",
                       reason="orchestrator only supports 2-leg entries today")
                return
            leg_y, leg_x = decision.legs
            res = place_two_legs(
                self.client,
                leg_y.coin, leg_y.is_buy, leg_y.size, leg_y.price,
                leg_x.coin, leg_x.is_buy, leg_x.size, leg_x.price,
                max_lag_sec=self.sg.max_one_leg_lag_sec,
                reduce_only=False,
                stop_loss_pct=decision.safety_stop_pct,
            )
            notify("entry", strategy=sname, ok=res.ok,
                   stop_y_oid=res.stop_y_oid, stop_x_oid=res.stop_x_oid,
                   **decision.telemetry)
            if res.ok and decision.state is not None:
                state = dict(decision.state)
                state["stop_y_oid"] = res.stop_y_oid
                state["stop_x_oid"] = res.stop_x_oid
                self.positions[sname] = state
                strat.on_filled(decision, {"ok": True, "stop_y_oid": res.stop_y_oid})

        elif decision.action == Action.EXIT:
            pos = self.positions[sname] or {}
            for coin, oid in (
                (decision.legs[0].coin, pos.get("stop_y_oid")),
                (decision.legs[1].coin, pos.get("stop_x_oid")),
            ):
                if oid:
                    self.client.cancel(coin, oid)
            leg_y, leg_x = decision.legs
            res = place_two_legs(
                self.client,
                leg_y.coin, leg_y.is_buy, leg_y.size, leg_y.price,
                leg_x.coin, leg_x.is_buy, leg_x.size, leg_x.price,
                max_lag_sec=self.sg.max_one_leg_lag_sec,
                reduce_only=True,
            )
            notify("exit", strategy=sname, reason=decision.reason, ok=res.ok,
                   **decision.telemetry)
            if res.ok:
                self.positions[sname] = None
                strat.on_filled(decision, {"ok": True})

    def iteration(self) -> None:
        if self.cb.halted:
            notify("halted", reason=self.cb.halt_reason)
            return

        all_coins = []
        for s in self.strategies:
            all_coins.extend(s.required_coins())

        try:
            # Enough history to seed all strategies' rolling windows.
            # StatArb needs hedge(960) + z(240) + ADF(960) + buffer.
            market_data = _build_market_data(self.client, all_coins, n_bars=2200)
            self.cb.record_success()
        except Exception as exc:
            self.cb.record_error(str(exc))
            notify("error", stage="fetch_market_data", exc=str(exc))
            return

        for strat in self.strategies:
            try:
                decision = strat.compute_signal(market_data, self.positions[strat.name])
                self._execute_decision(strat, decision)
            except Exception as exc:
                self.cb.record_error(str(exc))
                notify("error", strategy=strat.name, stage="compute_signal", exc=str(exc))

    def loop_forever(self, once: bool = False) -> None:
        notify("orchestrator_start", strategies=[s.name for s in self.strategies],
               network=self.client.network, dry_run=self.client.dry_run)
        if once:
            self.iteration()
            return
        while True:
            try:
                wait_ms = next_bar_close_ts() - int(time.time() * 1000)
                log.info(f"sleeping {wait_ms/1000:.1f}s until next bar")
                time.sleep(max(1, wait_ms / 1000) + 2)
                self.iteration()
            except KeyboardInterrupt:
                notify("orchestrator_stop", reason="KeyboardInterrupt")
                break
            except Exception as exc:
                self.cb.record_error(str(exc))
                notify("error", stage="loop", exc=str(exc))
                time.sleep(30)


def build_default_client() -> HLClient:
    """Robust HLClient init with retries for LibreSSL on macOS."""
    for attempt in range(10):
        try:
            return HLClient(network="testnet")
        except Exception as exc:
            log.warning(f"HLClient init failed attempt {attempt+1}/10: {exc}")
            time.sleep(min(5 * (attempt + 1), 60))
    raise RuntimeError("HLClient init failed after 10 attempts")
