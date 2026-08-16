"""Performance audit over the governance-owned evaluation database.

The raw signal database is an immutable input snapshot for governance. This
module reads finalized evaluations from nexus_governance.db and writes the
engine governance state plus the derived performance.json snapshot.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import KILL_THRESHOLD, PERFORMANCE_FILE, RECOVERY_THRESHOLD
from governance_db import (
    get_connection,
    init_governance_db,
    load_engine_status,
    upsert_engine_governance,
)


BASE_DIR = Path(__file__).resolve().parent
PERFORMANCE_PATH = BASE_DIR / PERFORMANCE_FILE
MAX_SIGNALS_PER_ENGINE = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | AUDIT | %(levelname)s | %(message)s",
)


def determine_status(
    previous_status: str,
    win_rate: float,
    total_trades: int,
) -> str:
    """Apply the LIVE/RECOVERY state machine after a minimum sample."""
    status = previous_status

    if total_trades < 5:
        return status

    if win_rate < KILL_THRESHOLD:
        return "RECOVERY"

    if previous_status == "RECOVERY" and win_rate >= RECOVERY_THRESHOLD:
        return "LIVE"

    return status


def load_engine_metrics() -> dict[str, dict[str, int]]:
    """Aggregate the most recent finalized evaluations by engine."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT engine_id, outcome
            FROM (
                SELECT
                    engine_id,
                    outcome,
                    ROW_NUMBER() OVER (
                        PARTITION BY engine_id
                        ORDER BY checked_at DESC
                    ) AS row_number
                FROM signal_evaluations
                WHERE outcome IN ('WIN', 'LOSS', 'PENDING', 'ERROR')
            )
            WHERE row_number <= ?
            ORDER BY engine_id
            """,
            (MAX_SIGNALS_PER_ENGINE,),
        ).fetchall()

    metrics: dict[str, dict[str, int]] = {}

    for row in rows:
        engine = str(row["engine_id"])
        metrics.setdefault(
            engine,
            {
                "wins": 0,
                "losses": 0,
                "pending": 0,
                "errors": 0,
            },
        )

        outcome = str(row["outcome"])
        if outcome == "WIN":
            metrics[engine]["wins"] += 1
        elif outcome == "LOSS":
            metrics[engine]["losses"] += 1
        elif outcome == "PENDING":
            metrics[engine]["pending"] += 1
        elif outcome == "ERROR":
            metrics[engine]["errors"] += 1

    return metrics


def write_performance_snapshot(performance: dict[str, dict]) -> None:
    """Atomically replace the derived performance JSON snapshot."""
    PERFORMANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = PERFORMANCE_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(performance, file, indent=4, sort_keys=True)
        file.write("\n")

    temp_path.replace(PERFORMANCE_PATH)


def run_audit() -> None:
    logging.info("--- STARTING PERFORMANCE AUDIT ---")
    init_governance_db()

    metrics = load_engine_metrics()

    if not metrics:
        logging.info("No governance evaluations are available for audit.")
        write_performance_snapshot({})
        return

    performance: dict[str, dict] = {}

    for engine, values in metrics.items():
        wins = int(values["wins"])
        losses = int(values["losses"])
        pending = int(values["pending"])
        errors = int(values["errors"])
        total = wins + losses
        win_rate = wins / total * 100 if total else 0.0

        previous_status = load_engine_status(engine) or "LIVE"
        status = determine_status(
            previous_status=previous_status,
            win_rate=win_rate,
            total_trades=total,
        )

        upsert_engine_governance(
            engine_id=engine,
            status=status,
            win_rate=win_rate,
            total_trades=total,
            wins=wins,
            losses=losses,
            pending=pending,
            errors=errors,
        )

        performance[engine] = {
            "win_rate": round(win_rate, 2),
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "errors": errors,
            "status": status,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

        logging.info(
            "Engine=%s | win_rate=%.2f%% | trades=%d | wins=%d | "
            "losses=%d | pending=%d | status=%s",
            engine,
            win_rate,
            total,
            wins,
            losses,
            pending,
            status,
        )

    write_performance_snapshot(performance)
    logging.info("Audit complete. Results saved to %s", PERFORMANCE_PATH)


if __name__ == "__main__":
    run_audit()
