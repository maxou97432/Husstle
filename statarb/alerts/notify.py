"""
Alert sink. Writes to a JSONL journal in all cases; additionally posts to
Telegram if TG_BOT_TOKEN and TG_CHAT_ID are set in env.
"""
from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("statarb.alerts")

JOURNAL_PATH = Path(__file__).parent.parent / "live_journal.jsonl"


def _telegram_send(text: str) -> None:
    tok = os.environ.get("TG_BOT_TOKEN", "").strip()
    chat = os.environ.get("TG_CHAT_ID", "").strip()
    if not tok or not chat:
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception as exc:
        logger.warning(f"telegram send failed: {exc}")


def notify(event: str, **payload: Any) -> None:
    """
    Log an event. Always writes to JSONL. Pings Telegram for high-priority
    events (fills, halts, errors).
    """
    rec = {"ts_ms": int(time.time() * 1000), "event": event, **payload}
    with JOURNAL_PATH.open("a") as f:
        f.write(json.dumps(rec) + "\n")

    if event in {"fill", "entry", "exit", "halt", "error", "kill_switch"}:
        text = f"*{event.upper()}* `{json.dumps(payload, default=str)[:300]}`"
        _telegram_send(text)
    logger.info(f"{event} {payload}")
