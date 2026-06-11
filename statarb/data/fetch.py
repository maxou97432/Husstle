import time
import httpx
from data.store import get_conn, upsert_candles, upsert_funding

HL_REST = "https://api.hyperliquid.xyz/info"
INTERVAL = "4h"
INTERVAL_MS = 4 * 3600 * 1000
FUNDING_INTERVAL_MS = 3600 * 1000
MAX_CANDLES_PER_PAGE = 500
MAX_FUNDING_PER_PAGE = 500


def _post(client: httpx.Client, payload: dict, retries: int = 5) -> dict:
    delay = 1.0
    for attempt in range(retries):
        try:
            r = client.post(HL_REST, json=payload, timeout=20)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, Exception) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def fetch_candles(coin: str, start_ms: int, end_ms: int, conn=None) -> list[dict]:
    rows: list[dict] = []
    cursor = start_ms
    with httpx.Client() as client:
        while cursor < end_ms:
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": INTERVAL,
                    "startTime": cursor,
                    "endTime": min(cursor + MAX_CANDLES_PER_PAGE * INTERVAL_MS, end_ms),
                },
            }
            data = _post(client, payload)
            if not data:
                break
            for c in data:
                rows.append({
                    "coin": coin,
                    "ts": int(c["t"]),
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                })
            last_ts = int(data[-1]["t"])
            if last_ts <= cursor:
                break
            cursor = last_ts + INTERVAL_MS
            time.sleep(0.1)

    if conn is not None:
        upsert_candles(conn, rows)
    return rows


def fetch_funding(coin: str, start_ms: int, end_ms: int, conn=None) -> list[dict]:
    rows: list[dict] = []
    cursor = start_ms
    with httpx.Client() as client:
        while cursor < end_ms:
            payload = {
                "type": "fundingHistory",
                "req": {
                    "coin": coin,
                    "startTime": cursor,
                    "endTime": min(cursor + MAX_FUNDING_PER_PAGE * FUNDING_INTERVAL_MS, end_ms),
                },
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


def fetch_all(coins: list[str], lookback_days: int = 365, testnet: bool = False) -> None:
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
