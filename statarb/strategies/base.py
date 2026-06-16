"""
Abstract Strategy class.

Each trading strategy implements this interface. The orchestrator polls one
or more Strategy instances each TF bar:

    bars passed → Strategy.compute_signal() → Decision
                  → Decision tells orchestrator: hold / enter / exit
                  → orchestrator places orders via core.execution

This keeps the orchestrator agnostic of the actual edge — it only knows how
to translate Decisions into HL orders, manage risk safeguards, and journal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    HOLD     = "hold"
    ENTER    = "enter"
    EXIT     = "exit"


@dataclass
class Leg:
    """A single order leg the orchestrator will place."""
    coin: str
    is_buy: bool
    size: float
    price: float            # for maker post-only; orchestrator may override
    tif: str = "Alo"        # post-only by default
    reduce_only: bool = False


@dataclass
class Decision:
    action: Action
    legs: list[Leg] = field(default_factory=list)
    reason: str = ""
    # Strategy-specific telemetry (z-score, ADF, funding, etc.) for journaling.
    telemetry: dict[str, Any] = field(default_factory=dict)
    # If action == ENTER, the orchestrator stores this as opaque state and
    # passes it back on subsequent compute_signal() calls via `position`.
    state: dict[str, Any] | None = None
    # Optional native stop-loss percentage attached at entry (per leg).
    safety_stop_pct: float | None = None


class Strategy(ABC):
    """Each concrete strategy implements the interface below.

    The orchestrator calls compute_signal() once per TF bar. The strategy
    is responsible for *both* entry and exit decisions; it inspects the
    current `position` (the state dict it returned at entry, or None).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short slug, e.g. 'statarb', 'funding_carry'."""

    @abstractmethod
    def required_coins(self) -> list[str]:
        """Coins the orchestrator must fetch market data for."""

    @abstractmethod
    def compute_signal(
        self,
        market_data: dict,        # {coin: {ts, close, beta, funding, ...}}
        position: dict | None,
    ) -> Decision:
        """Return what to do at this bar."""

    def on_filled(self, decision: Decision, fill_info: dict) -> None:
        """Optional hook — called after the orchestrator successfully fills entry/exit."""
        pass
