"""Generate the weekly performance report from governance-owned outcomes."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from config import GOVERNANCE_DB_FILE


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / GOVERNANCE_DB_FILE
REPORT_PATH = BASE_DIR / "weekly_performance_report.csv"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def load_weekly_evaluations() -> pd.DataFrame:
    """Load finalized evaluations from the last seven days."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Governance database not found: {DB_PATH}")

    start = datetime.now(timezone.utc) - timedelta(days=7)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        return pd.read_sql_query(
            """
            SELECT
                signal_id,
                engine_id,
                outcome,
                detected_at,
                checked_at,
                evaluation_version
            FROM signal_evaluations
            WHERE outcome IN ('WIN', 'LOSS')
              AND checked_at >= ?
            ORDER BY checked_at ASC
            """,
            conn,
            params=(start.isoformat(),),
        )
    finally:
        conn.close()


def send_discord_report(
    *,
    wins: int,
    losses: int,
    win_rate: float,
    engines: int,
) -> None:
    """Send a concise weekly governance summary to Discord."""
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ DISCORD_WEBHOOK missing. Skipping Discord notification.")
        return

    payload = {
        "username": "Nexus Performance Auditor",
        "embeds": [
            {
                "title": "🛡️ Nexus Weekly Governance Report",
                "fields": [
                    {
                        "name": "Weekly Win Rate",
                        "value": f"{win_rate:.1f}%",
                        "inline": True,
                    },
                    {
                        "name": "Finalized Outcomes",
                        "value": f"{wins} Wins / {losses} Losses",
                        "inline": True,
                    },
                    {
                        "name": "Engines Evaluated",
                        "value": str(engines),
                        "inline": True,
                    },
                ],
                "footer": {
                    "text": "Derived from Nexus governance evaluations",
                },
            }
        ],
    }

    response = requests.post(
        DISCORD_WEBHOOK_URL,
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
    print("✅ Weekly Discord report sent.")


def run_weekly_report() -> None:
    df = load_weekly_evaluations()

    if df.empty:
        print("⚠️ No finalized outcomes were recorded during the last seven days.")
        pd.DataFrame(
            columns=[
                "signal_id",
                "engine_id",
                "outcome",
                "detected_at",
                "checked_at",
                "evaluation_version",
            ]
        ).to_csv(REPORT_PATH, index=False)
        return

    wins = int((df["outcome"] == "WIN").sum())
    losses = int((df["outcome"] == "LOSS").sum())
    total = wins + losses
    win_rate = wins / total * 100 if total else 0.0
    engines = int(df["engine_id"].nunique())

    df.to_csv(REPORT_PATH, index=False)

    print(f"✅ Weekly report generated: {REPORT_PATH}")
    print(f"   Wins: {wins}")
    print(f"   Losses: {losses}")
    print(f"   Win rate: {win_rate:.2f}%")
    print(f"   Engines: {engines}")

    send_discord_report(
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        engines=engines,
    )


if __name__ == "__main__":
    run_weekly_report()
