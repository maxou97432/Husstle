import time
import httpx
from data.store import get_conn, upsert_candles, upsert_funding
from config import INTERVAL, INTERVAL_MS, LOOKBACK_DAYS_DEFAULT

HL_REST = "https://api.hyperliquid.xyz/info"
FUNDING_INTERVAL_MS = 3600 * 1000
MAX_CANDLES_PER_PAGE = 5000
MAX_FUNDING_PER_PAGE = 5000


def _post(client: httpx.Client, payload: dict, retries: int = 5) -> dict:
    delay = 1.0
    for attempt in range(retries):
        try:
            r = client.post(HL_REST, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, Exception):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return {}


def fetch_candles(coin: str, start_ms: int, end_ms: int, conn=None) -> list[dict]:
    """
    Fixed-step pagination forward. Each page covers MAX_CANDLES_PER_PAGE bars
    of calendar time. We advance by that span regardless of response size, so
    pre-listing empty pages don't abort the fetch and SSL hiccups don't cause
    silent gaps. PK ON CONFLICT DO NOTHING handles overlap.
    """
    rows: list[dict] = []
    page_span = MAX_CANDLES_PER_PAGE * INTERVAL_MS
    n_pages = max(1, (end_ms - start_ms + page_span - 1) // page_span)
    with httpx.Client() as client:
        for page in range(n_pages):
            cursor = start_ms + page * page_span
            if cursor >= end_ms:
                break
            page_end = min(cursor + page_span, end_ms)
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": INTERVAL,
                    "startTime": cursor,
                    "endTime": page_end,
                },
            }
            try:
                data = _post(client, payload)
            except Exception as exc:
                print(f"  page {page}/{n_pages-1} ERROR {type(exc).__name__}, continuing")
                time.sleep(2.0)
                continue
            for c in data or []:
                rows.append({
                    "coin": coin,
                    "ts": int(c["t"]),
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                })
            if data:
                print(f"  page {page}/{n_pages-1}: +{len(data):>5d} candles "
                      f"({(end_ms-cursor)/86400000:.0f}d ago → {(end_ms-page_end)/86400000:.0f}d ago)")
            time.sleep(0.15)

    if conn is not None:
        upsert_candles(conn, rows)
    return rows


def fetch_funding(coin: str, start_ms: int, end_ms: int, conn=None) -> list[dict]:
    """
    fundingHistory endpoint — no 'req' wrapper, coin at top level.
    """
    rows: list[dict] = []
    cursor = start_ms
    with httpx.Client() as client:
        while cursor < end_ms:
            page_end = min(cursor + MAX_FUNDING_PER_PAGE * FUNDING_INTERVAL_MS, end_ms)
            payload = {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": cursor,
                "endTime": page_end,
            }
            data = _post(client, payload)
            if not data:
                break
            for f in data:
                rows.append({
                    "coin": coin,
                    "ts": int(f["time"]),
                    "rate": float(f["fundingRate"]),
                })
            last_ts = int(data[-1]["time"])
            if last_ts <= cursor:
                break
            cursor = last_ts + FUNDING_INTERVAL_MS
            time.sleep(0.1)

    if conn is not None:
        upsert_funding(conn, rows)
    return rows


def fetch_all(coins: list[str], lookback_days: int = LOOKBACK_DAYS_DEFAULT, testnet: bool = False) -> None:
    global HL_REST
    if testnet:
        HL_REST = "https://api.hyperliquid-testnet.xyz/info"

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - lookback_days * 86400 * 1000

    conn = get_conn()
    for coin in coins:
        print(f"Fetching candles  {coin}…")
        fetch_candles(coin, start_ms, end_ms, conn)
        print(f"Fetching funding  {coin}…")
        fetch_funding(coin, start_ms, end_ms, conn)
    conn.close()
    print("Done.")
