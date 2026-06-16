"""
Thin Hyperliquid client wrapper.

- Defaults to TESTNET (the only network this phase is allowed to touch).
- Reads credentials from env: HL_PRIVATE_KEY, HL_ACCOUNT_ADDRESS.
- When credentials are missing OR dry_run=True, exposes a "paper" mode that
  fetches live market data but NEVER places orders. All would-be orders are
  logged with a [DRY] prefix.

The SDK signs orders via EIP-712 with the local key. Mainnet usage is gated
by an explicit `network="mainnet"` argument (default refuses it loudly).
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

logger = logging.getLogger("statarb.hl_client")


def _retry(fn, attempts: int = 4, base_delay: float = 0.5):
    """Retry transient SSL/network errors. LibreSSL on macOS is flaky."""
    delay = base_delay
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc).lower()
            transient = ("ssl" in msg or "timeout" in msg or "connection" in msg
                         or "decryption" in msg or "read" in msg)
            if not transient or i == attempts - 1:
                raise
            last_exc = exc
            time.sleep(delay)
            delay *= 2
    raise last_exc  # unreachable


@dataclass
class OrderResult:
    ok: bool
    filled_sz: float = 0.0
    filled_px: float = 0.0
    oid: int | None = None
    error: str = ""
    raw: dict | None = None


class HLClient:
    def __init__(
        self,
        network: str = "testnet",
        dry_run: bool | None = None,
    ):
        if network not in ("testnet", "mainnet"):
            raise ValueError("network must be 'testnet' or 'mainnet'")
        if network == "mainnet":
            # Brief §0: testnet only at this phase. Refuse mainnet wiring even if asked.
            raise RuntimeError(
                "Mainnet wiring is disabled in this phase per brief §0 — testnet only. "
                "If you really mean it, set the env STATARB_ALLOW_MAINNET=I_KNOW and rebuild this class."
            )
        self.network = network
        self.base_url = constants.TESTNET_API_URL

        self.priv_key = os.environ.get("HL_PRIVATE_KEY", "").strip()
        self.address = os.environ.get("HL_ACCOUNT_ADDRESS", "").strip()

        # dry_run defaults to True if no creds, False if creds present
        self.dry_run = (not self.priv_key) if dry_run is None else dry_run

        self.info = _retry(lambda: Info(self.base_url, skip_ws=True))
        self.exchange: Optional[Exchange] = None

        if not self.dry_run:
            from eth_account import Account
            wallet = Account.from_key(self.priv_key)
            self.exchange = Exchange(wallet, self.base_url, account_address=self.address or None)
            logger.info(f"HLClient LIVE ({network}) addr={self.address[:6]}…{self.address[-4:]}")
        else:
            logger.info(f"HLClient DRY-RUN ({network}) — no orders will be placed")

    # ── Market data (always live, no creds needed) ───────────────────────
    def mid_price(self, coin: str) -> float:
        mids = _retry(self.info.all_mids)
        return float(mids[coin])

    def book_top(self, coin: str) -> tuple[float, float]:
        """Returns (best_bid, best_ask)."""
        l2 = _retry(lambda: self.info.l2_snapshot(coin))
        bid = float(l2["levels"][0][0]["px"])
        ask = float(l2["levels"][1][0]["px"])
        return bid, ask

    def candles(self, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
        return _retry(lambda: self.info.candles_snapshot(coin, interval, start_ms, end_ms))

    def asset_meta(self, coin: str) -> dict:
        meta = _retry(self.info.meta)
        for asset in meta["universe"]:
            if asset["name"] == coin:
                return asset
        raise KeyError(f"coin {coin} not in HL universe")

    # ── Account state (live if creds, else None) ─────────────────────────
    def positions(self) -> dict:
        if self.dry_run or not self.address:
            return {}
        state = self.info.user_state(self.address)
        out = {}
        for p in state.get("assetPositions", []):
            pos = p["position"]
            out[pos["coin"]] = {
                "size": float(pos["szi"]),
                "entry_px": float(pos["entryPx"]) if pos.get("entryPx") else 0.0,
                "leverage": int(pos["leverage"]["value"]) if pos.get("leverage") else 0,
            }
        return out

    def margin_summary(self) -> dict:
        if self.dry_run or not self.address:
            return {"account_value": 0.0, "withdrawable": 0.0}
        state = self.info.user_state(self.address)
        ms = state.get("marginSummary", {})
        return {
            "account_value": float(ms.get("accountValue", 0)),
            "withdrawable": float(state.get("withdrawable", 0)),
        }

    # ── Trading ──────────────────────────────────────────────────────────
    def set_isolated_leverage(self, coin: str, leverage: int) -> OrderResult:
        if self.dry_run:
            logger.info(f"[DRY] set isolated leverage {coin} = {leverage}x")
            return OrderResult(ok=True)
        try:
            res = self.exchange.update_leverage(leverage, coin, is_cross=False)
            return OrderResult(ok=res.get("status") == "ok", raw=res)
        except Exception as exc:
            return OrderResult(ok=False, error=str(exc))

    def place_limit(
        self,
        coin: str,
        is_buy: bool,
        sz: float,
        px: float,
        tif: str = "Alo",   # post-only (maker). "Ioc" for taker, "Gtc" for resting.
        reduce_only: bool = False,
    ) -> OrderResult:
        side = "BUY" if is_buy else "SELL"
        if self.dry_run:
            logger.info(f"[DRY] {side} {sz} {coin} @ {px} ({tif}, reduce_only={reduce_only})")
            return OrderResult(ok=True, filled_sz=0.0, filled_px=px)
        try:
            res = self.exchange.order(coin, is_buy, sz, px, {"limit": {"tif": tif}}, reduce_only=reduce_only)
            status = res.get("response", {}).get("data", {}).get("statuses", [{}])[0]
            if "filled" in status:
                f = status["filled"]
                return OrderResult(ok=True, filled_sz=float(f["totalSz"]), filled_px=float(f["avgPx"]),
                                   oid=int(f.get("oid", 0)), raw=res)
            if "resting" in status:
                return OrderResult(ok=True, oid=int(status["resting"]["oid"]), raw=res)
            if "error" in status:
                return OrderResult(ok=False, error=status["error"], raw=res)
            return OrderResult(ok=True, raw=res)
        except Exception as exc:
            return OrderResult(ok=False, error=str(exc))

    def cancel(self, coin: str, oid: int) -> OrderResult:
        if self.dry_run:
            logger.info(f"[DRY] cancel {coin} oid={oid}")
            return OrderResult(ok=True)
        try:
            res = self.exchange.cancel(coin, oid)
            return OrderResult(ok=res.get("status") == "ok", raw=res)
        except Exception as exc:
            return OrderResult(ok=False, error=str(exc))
