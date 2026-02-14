import sqlite3
import ccxt
import requests
import logging
import time
import os
from datetime import datetime, timezone
from config import DB_FILE

# ===============================
# Environment / Config
# ===============================

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
SINGLE_RUN = os.getenv("SINGLE_RUN", "false").lower() == "true"

POLL_INTERVAL = 300
MAX_SIGNALS_PER_CYCLE = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, DB_FILE)

# ===============================
# Logging
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ALERT_MONITOR | %(levelname)s | %(message)s"
)

# ===============================
# Exchange
# ===============================

exchange = ccxt.gateio({
    "enableRateLimit": True,
    "timeout": 15000
})

# ===============================
# Database Schema
# ===============================

def ensure_schema(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            entry REAL,
            sl REAL,
            tp REAL,
            signal TEXT NOT NULL,
            status TEXT DEFAULT 'ACTIVE',
            ts TEXT,
            closed_at TEXT
        )
    """)
    conn.commit()

# ===============================
# Database Helpers
# ===============================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    return conn


def fetch_active_signals():
    if not os.path.exists(DB_PATH):
        logging.warning("Database file missing. No signals to monitor.")
        return []

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, asset, timeframe, entry, sl, tp, signal
            FROM signals
            WHERE status = 'ACTIVE'
            ORDER BY ts DESC
            LIMIT ?
        """, (MAX_SIGNALS_PER_CYCLE,))

        rows = cursor.fetchall()
        conn.close()
        return rows

    except Exception as e:
        logging.error(f"Failed to fetch signals: {e}")
        return []


def update_signal_status(signal_id, status):
    try:
        conn = get_connection()
        conn.execute("""
            UPDATE signals
            SET status = ?, closed_at = ?
            WHERE id = ?
        """, (
            status,
            datetime.now(timezone.utc).isoformat(),
            signal_id
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to update signal {signal_id}: {e}")

# ===============================
# Market Check Logic
# ===============================

def check_market_hit(asset, timeframe, sl, tp, signal_type):
    try:
        ohlcv = exchange.fetch_ohlcv(asset, timeframe, limit=50)
    except Exception as e:
        logging.error(f"Market fetch failed for {asset}: {e}")
        return None

    for candle in ohlcv:
        high = candle[2]
        low = candle[3]

        if signal_type.upper() == "LONG":
            if low <= sl:
                return "STOP_LOSS"
            if high >= tp:
                return "TAKE_PROFIT"

        elif signal_type.upper() == "SHORT":
            if high >= sl:
                return "STOP_LOSS"
            if low <= tp:
                return "TAKE_PROFIT"

    return None

# ===============================
# Notifications
# ===============================

def send_webhook(message):
    if not WEBHOOK_URL:
        logging.warning("No webhook configured.")
        return

    try:
        requests.post(
            WEBHOOK_URL,
            json={"text": message},
            timeout=10
        )
    except Exception as e:
        logging.error(f"Webhook failed: {e}")

# ===============================
# Monitor Cycle
# ===============================

def monitor_cycle():
    signals = fetch_active_signals()

    if not signals:
        logging.info("No active signals.")
        return

    logging.info(f"Monitoring {len(signals)} active signals.")

    for signal in signals:
        signal_id, asset, timeframe, entry, sl, tp, signal_type = signal

        result = check_market_hit(asset, timeframe, sl, tp, signal_type)

        if result:
            logging.info(f"{asset} {timeframe} hit {result}")

            update_signal_status(signal_id, result)

            message = (
                f"📊 {asset} ({timeframe})\n"
                f"Signal: {signal_type}\n"
                f"Result: {result}\n"
                f"Time: {datetime.now(timezone.utc).isoformat()}"
            )

            send_webhook(message)

# ===============================
# Main Loop
# ===============================

def run_monitor():
    logging.info("Starting alert monitor")

    while True:
        monitor_cycle()

        if SINGLE_RUN:
            logging.info("Single run mode enabled. Exiting.")
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_monitor()
