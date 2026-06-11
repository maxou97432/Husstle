import duckdb
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "statarb.duckdb"


def get_conn(path: str | Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(path))
    _init_schema(conn)
    return conn


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            coin    VARCHAR NOT NULL,
            ts      BIGINT  NOT NULL,
            open    DOUBLE  NOT NULL,
            high    DOUBLE  NOT NULL,
            low     DOUBLE  NOT NULL,
            close   DOUBLE  NOT NULL,
            volume  DOUBLE  NOT NULL,
            PRIMARY KEY (coin, ts)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding (
            coin    VARCHAR NOT NULL,
            ts      BIGINT  NOT NULL,
            rate    DOUBLE  NOT NULL,
            PRIMARY KEY (coin, ts)
        )
    """)


def upsert_candles(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO candles (coin, ts, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [(r["coin"], r["ts"], r["open"], r["high"], r["low"], r["close"], r["volume"]) for r in rows],
    )


def upsert_funding(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO funding (coin, ts, rate)
        VALUES (?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        [(r["coin"], r["ts"], r["rate"]) for r in rows],
    )


def load_candles(conn: duckdb.DuckDBPyConnection, coin: str) -> list[dict]:
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE coin=? ORDER BY ts",
        [coin],
    ).fetchall()
    cols = ["ts", "open", "high", "low", "close", "volume"]
    return [dict(zip(cols, r)) for r in rows]


def load_funding(conn: duckdb.DuckDBPyConnection, coin: str) -> list[dict]:
    rows = conn.execute(
        "SELECT ts, rate FROM funding WHERE coin=? ORDER BY ts",
        [coin],
    ).fetchall()
    return [{"ts": r[0], "rate": r[1]} for r in rows]
