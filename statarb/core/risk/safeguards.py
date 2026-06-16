"""
Pre-trade and during-trade safety checks.

These are the "do not deploy something dangerous" guards. The strategy's
own kill-switch (ADF) lives in backtest/engine.py and is replicated in
live_bot.py for the live signal loop.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Safeguards:
    max_account_drawdown_pct: float = 0.10   # halt new entries if equity drops 10% from peak
    max_one_leg_lag_sec: int = 5             # one-leg-filled-other-not tolerance
    max_consecutive_errors: int = 3          # halt on persistent broker errors
    max_open_pairs: int = 1                  # phase-1 scope: BTC/ETH only


class CircuitBreaker:
    """Tracks consecutive errors and global halt state."""
    def __init__(self, sg: Safeguards):
        self.sg = sg
        self.consecutive_errors = 0
        self.halted = False
        self.halt_reason = ""

    def record_error(self, reason: str) -> bool:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.sg.max_consecutive_errors:
            self.halt(f"{self.sg.max_consecutive_errors} consecutive errors — last: {reason}")
        return self.halted

    def record_success(self) -> None:
        self.consecutive_errors = 0

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def check_drawdown(self, peak_equity: float, current_equity: float) -> None:
        if peak_equity <= 0:
            return
        dd = (current_equity - peak_equity) / peak_equity
        if dd <= -self.sg.max_account_drawdown_pct:
            self.halt(f"account drawdown {dd:.1%} <= -{self.sg.max_account_drawdown_pct:.0%}")
