"""Persistence helpers for the pipeline-owned signals database.

This module is used by signal-generation engines only. Governance code must
never import this module to update signal lifecycle state.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


REQUIRED_COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "engine_id": "TEXT NOT NULL",
    "symbol": "TEXT NOT NULL",
    "timeframe": "TEXT NOT NULL",
    "direction": "TEXT NOT NULL",
    "entry": "REAL NOT NULL",
    "stop_loss": "REAL NOT NULL",
    "take_profit": "REAL NOT NULL",
    "confidence": "REAL",
    "rsi": "REAL",
    "vol_change": "REAL",
    "dist_ema": "REAL",
    "reason": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'ACTIVE'",
    "timestamp": "TEXT NOT NULL",
}


@contextmanager
def get_connection(db_name: str) -> Iterator[sqlite3.Connection]:
    """Open a pipeline database connection with safe SQLite settings."""
    path = Path(db_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, timeout=30)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _table_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("PRAGMA table_info(signals)").fetchall()
    return [str(row[1]) for row in rows]


def _archive_legacy_table(conn: sqlite3.Connection) -> None:
    """Preserve an incompatible legacy table instead of destroying it."""
    suffix = 1
    while True:
        table_name = f"signals_legacy_{suffix}"
        exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        ).fetchone()

        if exists is None:
            break

        suffix += 1

    logging.warning(
        "Archiving incompatible legacy signals table as %s",
        table_name,
    )

    conn.execute(
        f'ALTER TABLE signals RENAME TO "{table_name}"'
    )


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            confidence REAL,
            rsi REAL,
            vol_change REAL,
            dist_ema REAL,
            reason TEXT,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            timestamp TEXT NOT NULL
        )
        """
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_signals_engine_timestamp "
        "ON signals(engine_id, timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_signals_status_timestamp "
        "ON signals(status, timestamp)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_signals_symbol_timeframe "
        "ON signals(symbol, timeframe, timestamp)"
    )


def _migrate_current_schema(conn: sqlite3.Connection) -> None:
    """Migrate a compatible schema without dropping signal history."""
    columns = set(_table_columns(conn))

    if not columns:
        _create_schema(conn)
        return

    if "engine_id" not in columns:
        _archive_legacy_table(conn)
        _create_schema(conn)
        return

    missing = set(REQUIRED_COLUMNS) - columns

    # SQLite cannot add a NOT NULL column without a constant default. The only
    # missing production column expected from older Nexus versions is status.
    if "status" in missing:
        conn.execute(
            "ALTER TABLE signals ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'"
        )
        missing.remove("status")

    if missing:
        raise RuntimeError(
            "Signal database schema is missing unsupported required columns: "
            + ", ".join(sorted(missing))
        )

    _create_schema(conn)


def init_db(db_name: str) -> None:
    """Initialize or safely migrate the pipeline-owned signal database."""
    with get_connection(db_name) as conn:
        _migrate_current_schema(conn)


def insert_signal(db_name: str, data_dict: Mapping[str, Any]) -> int:
    """Insert a validated signal and return its database ID."""
    if not data_dict:
        raise ValueError("Signal payload cannot be empty")

    payload = dict(data_dict)
    payload.setdefault("status", "ACTIVE")

    required = {
        "engine_id",
        "symbol",
        "timeframe",
        "direction",
        "entry",
        "stop_loss",
        "take_profit",
        "timestamp",
    }

    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(
            "Signal payload is missing required fields: "
            + ", ".join(missing)
        )

    columns = list(payload.keys())
    placeholders = ", ".join("?" for _ in columns)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)

    query = (
        f"INSERT INTO signals ({quoted_columns}) "
        f"VALUES ({placeholders})"
    )

    with get_connection(db_name) as conn:
        cursor = conn.execute(
            query,
            tuple(payload[column] for column in columns),
        )
        return int(cursor.lastrowid)
