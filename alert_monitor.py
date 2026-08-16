import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import requests

from config import DB_FILE, EXCHANGE_ID


# ============================================================
# Environment / Configuration
# ============================================================

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
SINGLE_RUN = os.getenv("SINGLE_RUN", "false").lower() == "true"

POLL_INTERVAL = 300
MAX_SIGNALS_PER_CYCLE = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, DB_FILE)

# Prevent processing candles that existed before the signal.
# A small tolerance is useful because exchange timestamps are
# normally candle-open timestamps.
SIGNAL_TIME_TOLERANCE_MS = 1_000


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ALERT_MONITOR | %(levelname)s | %(message)s",
)


# ============================================================
# Exchange Integration
# ============================================================

if EXCHANGE_ID not in ccxt.exchanges:
    raise EnvironmentError(
        f"Exchange '{EXCHANGE_ID}' is not supported by the "
        f"currently installed CCXT version."
    )

exchange_class = getattr(ccxt, EXCHANGE_ID)

exchange = exchange_class(
    {
        "enableRateLimit": True,
        "timeout": 15_000,
    }
)


# ============================================================
# Database Helpers
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Create a SQLite connection.

    A timeout prevents immediate failure if another process has
    a temporary SQLite lock.
    """
    return sqlite3.connect(
        DB_PATH,
        timeout=30,
    )


def fetch_active_signals() -> list[tuple]:
    """
    Fetch currently active signals.

    The timestamp is included because the market monitor must not
    evaluate candles that occurred before the signal was created.
    """
    if not os.path.exists(DB_PATH):
        logging.warning(
            "Database file missing. No signals to monitor."
        )
        return []

    conn = None

    try:
        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                symbol,
                timeframe,
                entry,
                stop_loss,
                take_profit,
                direction,
                timestamp
            FROM signals
            WHERE status = 'ACTIVE'
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (MAX_SIGNALS_PER_CYCLE,),
        )

        return cursor.fetchall()

    except sqlite3.Error as exc:
        logging.error(
            "Failed to fetch active signals: %s",
            exc,
        )
        return []

    finally:
        if conn is not None:
            conn.close()


def update_signal_status(
    signal_id: int,
    status: str,
) -> None:
    """
    Update a signal's lifecycle status.
    """
    conn = None

    try:
        conn = get_connection()

        conn.execute(
            """
            UPDATE signals
            SET status = ?
            WHERE id = ?
            """,
            (
                status,
                signal_id,
            ),
        )

        conn.commit()

    except sqlite3.Error as exc:
        logging.error(
            "Failed to update signal %s: %s",
            signal_id,
            exc,
        )

    finally:
        if conn is not None:
            conn.close()


# ============================================================
# Timestamp Helpers
# ============================================================

def parse_signal_timestamp(timestamp: str) -> Optional[int]:
    """
    Convert a database timestamp into milliseconds since epoch.

    Supports ISO-8601 timestamps and common SQLite datetime values.
    """
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
        logging.warning(
            "Unable to parse signal timestamp: %s",
            timestamp,
        )
        return None


# ============================================================
# Market Check Logic
# ============================================================

def check_market_hit(
    symbol: str,
    timeframe: str,
    signal_timestamp: str,
    stop_loss: float,
    take_profit: float,
    direction: str,
) -> Optional[str]:
    """
    Determine whether TP or SL has been reached after the signal
    was created.

    Important limitation:
    OHLC candles do not reveal intrabar ordering when both TP and
    SL occur inside the same candle. In that situation this
    function deliberately returns PENDING rather than inventing
    an execution order.
    """
    try:
        signal_ts_ms = parse_signal_timestamp(signal_timestamp)

        if signal_ts_ms is None:
            return None

        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            limit=100,
        )

    except Exception as exc:
        logging.error(
            "Market fetch failed for %s: %s",
            symbol,
            exc,
        )
        return None

    direction_upper = direction.upper()

    for candle in ohlcv:
        candle_timestamp = candle[0]
        high = float(candle[2])
        low = float(candle[3])

        # Ignore candles that existed before the signal.
        if candle_timestamp + SIGNAL_TIME_TOLERANCE_MS < signal_ts_ms:
            continue

        if direction_upper == "LONG":

            hit_stop = low <= float(stop_loss)
            hit_target = high >= float(take_profit)

        elif direction_upper == "SHORT":

            hit_stop = high >= float(stop_loss)
            hit_target = low <= float(take_profit)

        else:
            logging.error(
                "Unknown signal direction '%s' for %s",
                direction,
                symbol,
            )
            return None

        # Both levels were inside the same candle.
        #
        # OHLCV does not provide the sequence of intrabar events,
        # therefore do not fabricate a result.
        if hit_stop and hit_target:
            logging.warning(
                "Ambiguous candle for signal %s: both SL and TP "
                "were reached in the same OHLC candle.",
                symbol,
            )
            return None

        if hit_stop:
            return "STOP_LOSS"

        if hit_target:
            return "TAKE_PROFIT"

    return None


# ============================================================
# Notifications
# ============================================================

def send_webhook(message: str) -> None:
    """
    Send a notification to the configured Discord webhook.
    """
    if not WEBHOOK_URL:
        logging.warning(
            "WEBHOOK_URL is not configured."
        )
        return

    try:
        response = requests.post(
            WEBHOOK_URL,
            json={
                "content": message,
            },
            timeout=10,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        logging.error(
            "Webhook failed: %s",
            exc,
        )


# ============================================================
# Monitor Cycle
# ============================================================

def monitor_cycle() -> None:
    """
    Process all currently active signals.
    """
    signals = fetch_active_signals()

    if not signals:
        logging.info("No active signals.")
        return

    logging.info(
        "Monitoring %d active signal(s).",
        len(signals),
    )

    for signal in signals:
        (
            signal_id,
            symbol,
            timeframe,
            entry,
            stop_loss,
            take_profit,
            direction,
            signal_timestamp,
        ) = signal

        try:
            result = check_market_hit(
                symbol=symbol,
                timeframe=timeframe,
                signal_timestamp=signal_timestamp,
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                direction=direction,
            )

            if not result:
                continue

            logging.info(
                "%s %s signal %s hit %s",
                symbol,
                timeframe,
                signal_id,
                result,
            )

            update_signal_status(
                signal_id,
                result,
            )

            message = (
                f"📊 **{symbol}** ({timeframe})\n"
                f"**Signal:** {direction}\n"
                f"**Entry:** {entry}\n"
                f"**Stop Loss:** {stop_loss}\n"
                f"**Take Profit:** {take_profit}\n"
                f"**Result:** {result}\n"
                f"**Signal Time:** {signal_timestamp}\n"
                f"**Detected:** "
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )

            send_webhook(message)

        except Exception as exc:
            logging.exception(
                "Unexpected error processing signal %s: %s",
                signal_id,
                exc,
            )


# ============================================================
# Main Loop
# ============================================================

def run_monitor() -> None:
    """
    Run the alert monitor continuously or once, depending on
    SINGLE_RUN.
    """
    logging.info(
        "Starting alert monitor | single_run=%s",
        SINGLE_RUN,
    )

    while True:
        monitor_cycle()

        if SINGLE_RUN:
            logging.info(
                "Single run mode enabled. Exiting."
            )
            break

        time.sleep(POLL_INTERVAL)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    run_monitor()
