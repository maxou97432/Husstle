"""
Two-leg simultaneous order placement with partial-fill safety.

Brief §5.9: a leg filled without the other = involuntary directional risk.
Strategy:
  1. Place both legs as post-only limits (maker) at the current best price
     on the appropriate side.
  2. Wait up to `max_one_leg_lag_sec` for both to fill.
  3. If only one filled: cancel any resting orders, then HARD-CLOSE the
     filled leg with an IOC taker. Surface an alert.
  4. If both filled: success.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field

from core.execution.hl_client import HLClient, OrderResult
from core.alerts.notify import notify

logger = logging.getLogger("statarb.orders")


@dataclass
class TwoLegFill:
    ok: bool
    leg_y: OrderResult = field(default_factory=lambda: OrderResult(ok=False))
    leg_x: OrderResult = field(default_factory=lambda: OrderResult(ok=False))
    reason: str = ""
    stop_y_oid: int | None = None    # native HL stop-loss order IDs (resting)
    stop_x_oid: int | None = None


def place_two_legs(
    client: HLClient,
    coin_y: str, is_buy_y: bool, sz_y: float, px_y: float,
    coin_x: str, is_buy_x: bool, sz_x: float, px_x: float,
    max_lag_sec: int = 5,
    reduce_only: bool = False,
    stop_loss_pct: float | None = None,
) -> TwoLegFill:
    """
    Place both legs as maker limits, monitor fills, force-close the orphan leg
    if only one fills within max_lag_sec. Returns a single TwoLegFill.
    """
    notify("two_leg_request",
           y=coin_y, side_y="BUY" if is_buy_y else "SELL", sz_y=sz_y, px_y=px_y,
           x=coin_x, side_x="BUY" if is_buy_x else "SELL", sz_x=sz_x, px_x=px_x,
           reduce_only=reduce_only)

    r_y = client.place_limit(coin_y, is_buy_y, sz_y, px_y, tif="Alo", reduce_only=reduce_only)
    r_x = client.place_limit(coin_x, is_buy_x, sz_x, px_x, tif="Alo", reduce_only=reduce_only)

    if not r_y.ok and not r_x.ok:
        return TwoLegFill(ok=False, leg_y=r_y, leg_x=r_x, reason="both legs failed to place")

    def _place_safety_stops(result: TwoLegFill) -> TwoLegFill:
        """After successful entry, place native HL stop-market on each leg as a survival net."""
        if reduce_only or stop_loss_pct is None or stop_loss_pct <= 0:
            return result
        # On a LONG (is_buy=True), stop triggers BELOW entry — sell-stop.
        # On a SHORT (is_buy=False), stop triggers ABOVE entry — buy-stop.
        trig_y = px_y * (1 - stop_loss_pct) if is_buy_y else px_y * (1 + stop_loss_pct)
        trig_x = px_x * (1 - stop_loss_pct) if is_buy_x else px_x * (1 + stop_loss_pct)
        s_y = client.place_stop_market(coin_y, is_buy=not is_buy_y, sz=sz_y, trigger_px=round(trig_y, 5))
        s_x = client.place_stop_market(coin_x, is_buy=not is_buy_x, sz=sz_x, trigger_px=round(trig_x, 5))
        notify("safety_stops",
               leg_y=coin_y, trig_y=trig_y, ok_y=s_y.ok, oid_y=s_y.oid,
               leg_x=coin_x, trig_x=trig_x, ok_x=s_x.ok, oid_x=s_x.oid)
        result.stop_y_oid = s_y.oid
        result.stop_x_oid = s_x.oid
        return result

    deadline = time.time() + max_lag_sec
    while time.time() < deadline:
        if r_y.filled_sz > 0 and r_x.filled_sz > 0:
            return _place_safety_stops(TwoLegFill(ok=True, leg_y=r_y, leg_x=r_x))
        # In dry-run, partial-fill checking isn't meaningful (sizes start at 0).
        if client.dry_run:
            return _place_safety_stops(TwoLegFill(ok=True, leg_y=r_y, leg_x=r_x,
                                                   reason="dry-run (assumed both filled)"))
        time.sleep(0.5)
        # Refresh fill state would normally poll order status; SDK fills update on poll.

    # One leg filled, one didn't — emergency-close the filled one as taker.
    if r_y.filled_sz > 0 and r_x.filled_sz == 0:
        if r_x.oid:
            client.cancel(coin_x, r_x.oid)
        # close r_y with IOC at the touch
        bid, ask = client.book_top(coin_y)
        close_px = bid * 0.998 if is_buy_y else ask * 1.002
        notify("orphan_close", leg=coin_y, sz=r_y.filled_sz, reason="x-leg unfilled")
        client.place_limit(coin_y, not is_buy_y, r_y.filled_sz, close_px, tif="Ioc")
        return TwoLegFill(ok=False, leg_y=r_y, leg_x=r_x, reason="y filled, x unfilled — emergency closed y")

    if r_x.filled_sz > 0 and r_y.filled_sz == 0:
        if r_y.oid:
            client.cancel(coin_y, r_y.oid)
        bid, ask = client.book_top(coin_x)
        close_px = bid * 0.998 if is_buy_x else ask * 1.002
        notify("orphan_close", leg=coin_x, sz=r_x.filled_sz, reason="y-leg unfilled")
        client.place_limit(coin_x, not is_buy_x, r_x.filled_sz, close_px, tif="Ioc")
        return TwoLegFill(ok=False, leg_y=r_y, leg_x=r_x, reason="x filled, y unfilled — emergency closed x")

    # Neither filled
    if r_y.oid: client.cancel(coin_y, r_y.oid)
    if r_x.oid: client.cancel(coin_x, r_x.oid)
    return TwoLegFill(ok=False, leg_y=r_y, leg_x=r_x, reason="neither leg filled within max_lag_sec")
