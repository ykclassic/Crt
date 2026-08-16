"""Governance-owned persistence for signal evaluations and engine state.

This module deliberately never writes to the pipeline-owned signals database.
The raw signal database is an input snapshot; all lifecycle/evaluation state is
stored here instead.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from config import GOVERNANCE_DB_FILE


BASE_DIR = Path(__file__).resolve().parent
GOVERNANCE_DB_PATH = BASE_DIR / GOVERNANCE_DB_FILE

SCHEMA_VERSION = 1


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a configured governance database connection."""
    conn = sqlite3.connect(GOVERNANCE_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
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


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def init_governance_db() -> None:
    """Create the governance schema without touching the signal database."""
    GOVERNANCE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signal_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL UNIQUE,
                engine_id TEXT NOT NULL,
                outcome TEXT NOT NULL CHECK (
                    outcome IN ('PENDING', 'WIN', 'LOSS', 'ERROR')
                ),
                detected_at TEXT,
                checked_at TEXT NOT NULL,
                evaluation_version INTEGER NOT NULL DEFAULT 1,
                evidence TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_signal_evaluations_engine
                ON signal_evaluations(engine_id);

            CREATE INDEX IF NOT EXISTS idx_signal_evaluations_outcome
                ON signal_evaluations(outcome);

            CREATE INDEX IF NOT EXISTS idx_signal_evaluations_checked
                ON signal_evaluations(checked_at);

            CREATE TABLE IF NOT EXISTS engine_governance (
                engine_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK (
                    status IN ('LIVE', 'RECOVERY', 'DISABLED')
                ),
                win_rate REAL NOT NULL DEFAULT 0,
                total_trades INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0,
                losses INTEGER NOT NULL DEFAULT 0,
                pending INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_engine_governance_status
                ON engine_governance(status);
            """
        )

        conn.execute(
            """
            INSERT INTO schema_metadata(key, value)
            VALUES('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(SCHEMA_VERSION),),
        )


def record_evaluation(
    *,
    signal_id: int,
    engine_id: str,
    outcome: str,
    detected_at: Optional[str],
    evidence: Optional[str] = None,
) -> None:
    """Insert or update the governance evaluation for one signal."""
    outcome = outcome.upper()
    if outcome not in {"PENDING", "WIN", "LOSS", "ERROR"}:
        raise ValueError(f"Unsupported governance outcome: {outcome}")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO signal_evaluations(
                signal_id,
                engine_id,
                outcome,
                detected_at,
                checked_at,
                evaluation_version,
                evidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
                engine_id = excluded.engine_id,
                outcome = excluded.outcome,
                detected_at = excluded.detected_at,
                checked_at = excluded.checked_at,
                evaluation_version = excluded.evaluation_version,
                evidence = excluded.evidence
            """,
            (
                signal_id,
                engine_id,
                outcome,
                detected_at,
                utc_now(),
                SCHEMA_VERSION,
                evidence,
            ),
        )


def get_terminal_signal_ids() -> set[int]:
    """Return signals that have a final WIN/LOSS evaluation."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT signal_id
            FROM signal_evaluations
            WHERE outcome IN ('WIN', 'LOSS')
            """
        ).fetchall()

    return {int(row["signal_id"]) for row in rows}


def upsert_engine_governance(
    *,
    engine_id: str,
    status: str,
    win_rate: float,
    total_trades: int,
    wins: int,
    losses: int,
    pending: int,
    errors: int,
) -> None:
    """Persist the current governance state for an engine."""
    status = status.upper()
    if status not in {"LIVE", "RECOVERY", "DISABLED"}:
        raise ValueError(f"Unsupported governance status: {status}")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO engine_governance(
                engine_id,
                status,
                win_rate,
                total_trades,
                wins,
                losses,
                pending,
                errors,
                last_updated
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_id) DO UPDATE SET
                status = excluded.status,
                win_rate = excluded.win_rate,
                total_trades = excluded.total_trades,
                wins = excluded.wins,
                losses = excluded.losses,
                pending = excluded.pending,
                errors = excluded.errors,
                last_updated = excluded.last_updated
            """,
            (
                engine_id,
                status,
                round(float(win_rate), 2),
                int(total_trades),
                int(wins),
                int(losses),
                int(pending),
                int(errors),
                utc_now(),
            ),
        )


def load_engine_status(engine_id: str) -> Optional[str]:
    """Return the previously persisted status for an engine."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT status
            FROM engine_governance
            WHERE engine_id = ?
            """,
            (engine_id,),
        ).fetchone()

    return str(row["status"]) if row else None


if __name__ == "__main__":
    init_governance_db()
    print(f"Governance database initialized: {GOVERNANCE_DB_PATH}")
