"""Evaluate pipeline-generated signals without mutating the signal database.

Ownership contract:
    nexus_signals.db     -> pipeline-owned, read-only here
    nexus_governance.db  -> governance-owned, writable here
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import requests

from config import DB_FILE, EXCHANGE_ID
from governance_db import get_terminal_signal_ids, init_governance_db, record_evaluation


WEBHOOK_URL = os.getenv("WEBHOOK_URL")
SINGLE_RUN = os.getenv("SINGLE_RUN", "false").lower() == "true"
POLL_INTERVAL = 300
MAX_SIGNALS_PER_CYCLE = 50
SIGNAL_TIME_TOLERANCE_MS = 1_000

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / DB_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ALERT_MONITOR | %(levelname)s | %(message)s",
)


if EXCHANGE_ID not in ccxt.exchanges:
    raise EnvironmentError(
        f"Exchange '{EXCHANGE_ID}' is not supported by the installed CCXT version."
    )

exchange = getattr(ccxt, EXCHANGE_ID)(
    {
        "enableRateLimit": True,
        "timeout": 15_000,
    }
)


def get_signal_connection() -> sqlite3.Connection:
    """Open the pipeline-owned signal database in read-only mode."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Signal database not found: {DB_PATH}")

    uri = f"file:{DB_PATH.resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30)


def fetch_unresolved_signals() -> list[sqlite3.Row]:
    """Fetch signals that do not yet have a terminal WIN/LOSS evaluation."""
    terminal_ids = get_terminal_signal_ids()
    conn = None

    try:
        conn = get_signal_connection()
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                id,
                engine_id,
                symbol,
                timeframe,
                entry,
                stop_loss,
                take_profit,
                direction,
                timestamp
            FROM signals
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (MAX_SIGNALS_PER_CYCLE * 3,),
        ).fetchall()

        unresolved = [
            row for row in rows
            if int(row["id"]) not in terminal_ids
        ]
        return unresolved[:MAX_SIGNALS_PER_CYCLE]
    finally:
        if conn is not None:
            conn.close()


def parse_signal_timestamp(timestamp: str) -> Optional[int]:
    """Convert an ISO-8601 or SQLite timestamp to epoch milliseconds."""
    if not timestamp:
        return None

    try:
        normalized = timestamp.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return int(dt.timestamp() * 1000)
    except ValueError:
        logging.warning("Unable to parse signal timestamp: %s", timestamp)
        return None


def check_market_hit(
    *,
    symbol: str,
    timeframe: str,
    signal_timestamp: str,
    stop_loss: float,
    take_profit: float,
    direction: str,
) -> Optional[str]:
    """Evaluate post-signal candles without fabricating intrabar order."""
    signal_ts_ms = parse_signal_timestamp(signal_timestamp)
    if signal_ts_ms is None:
        return None

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
    except Exception as exc:
        logging.error("Market fetch failed for %s: %s", symbol, exc)
        return None

    direction_upper = direction.upper()

    for candle in ohlcv:
        candle_timestamp = int(candle[0])
        high = float(candle[2])
        low = float(candle[3])

        if candle_timestamp + SIGNAL_TIME_TOLERANCE_MS < signal_ts_ms:
            continue

        if direction_upper == "LONG":
            hit_stop = low <= float(stop_loss)
            hit_target = high >= float(take_profit)
        elif direction_upper == "SHORT":
            hit_stop = high >= float(stop_loss)
            hit_target = low <= float(take_profit)
        else:
            logging.error("Unknown signal direction '%s'", direction)
            return None

        if hit_stop and hit_target:
            logging.warning(
                "Ambiguous candle for signal timestamp %s: both SL and TP were reached.",
                signal_timestamp,
            )
            return None

        if hit_stop:
            return "LOSS"

        if hit_target:
            return "WIN"

    return None


def send_webhook(message: str) -> None:
    """Send a Discord notification when configured."""
    if not WEBHOOK_URL:
        logging.warning("WEBHOOK_URL is not configured.")
        return

    try:
        response = requests.post(
            WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.error("Webhook failed: %s", exc)


def monitor_cycle() -> None:
    """Evaluate unresolved signals and persist only governance state."""
    init_governance_db()
    signals = fetch_unresolved_signals()

    if not signals:
        logging.info("No unresolved signals.")
        return

    logging.info("Evaluating %d unresolved signal(s).", len(signals))

    for signal in signals:
        signal_id = int(signal["id"])
        outcome = check_market_hit(
            symbol=str(signal["symbol"]),
            timeframe=str(signal["timeframe"]),
            signal_timestamp=str(signal["timestamp"]),
            stop_loss=float(signal["stop_loss"]),
            take_profit=float(signal["take_profit"]),
            direction=str(signal["direction"]),
        )

        if outcome is None:
            record_evaluation(
                signal_id=signal_id,
                engine_id=str(signal["engine_id"]),
                outcome="PENDING",
                detected_at=None,
                evidence="No unambiguous TP/SL event detected in available OHLCV data.",
            )
            continue

        detected_at = datetime.now(timezone.utc).isoformat()
        record_evaluation(
            signal_id=signal_id,
            engine_id=str(signal["engine_id"]),
            outcome=outcome,
            detected_at=detected_at,
            evidence="Outcome determined from post-signal OHLCV candles.",
        )

        message = (
            f"📊 **{signal['symbol']}** ({signal['timeframe']})\n"
            f"**Engine:** {signal['engine_id']}\n"
            f"**Signal:** {signal['direction']}\n"
            f"**Entry:** {signal['entry']}\n"
            f"**Stop Loss:** {signal['stop_loss']}\n"
            f"**Take Profit:** {signal['take_profit']}\n"
            f"**Result:** {outcome}\n"
            f"**Signal Time:** {signal['timestamp']}\n"
            f"**Detected:** {detected_at}"
        )
        send_webhook(message)


def run_monitor() -> None:
    logging.info("Starting alert monitor | single_run=%s", SINGLE_RUN)

    while True:
        try:
            monitor_cycle()
        except Exception as exc:
            logging.exception("Alert monitor cycle failed: %s", exc)

        if SINGLE_RUN:
            logging.info("Single run mode enabled. Exiting.")
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_monitor()
