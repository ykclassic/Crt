import sqlite3
import ccxt
import requests
import logging
import time
import os
from datetime import datetime, timezone
from config import DB_FILE, EXCHANGE_ID

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
# Exchange Integration
# ===============================

# Validate exchange support in the current environment before execution
if EXCHANGE_ID not in ccxt.exchanges:
    raise EnvironmentError(
        f"Exchange '{EXCHANGE_ID}' is not supported by the currently installed CCXT version."
    )

exchange = getattr(ccxt, EXCHANGE_ID)({
    "enableRateLimit": True,
    "timeout": 15000
})

# ===============================
# Database Helpers
# ===============================

def get_connection():
    # We strictly connect to the existing DB. Schema generation is handled by the core engines.
    return sqlite3.connect(DB_PATH)


def fetch_active_signals():
    if not os.path.exists(DB_PATH):
        logging.warning("Database file missing. No signals to monitor.")
        return []

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Query updated to match the unified nexus_signals.db schema
        cursor.execute("""
            SELECT id, symbol, timeframe, entry, stop_loss, take_profit, direction
            FROM signals
            WHERE status = 'ACTIVE'
            ORDER BY timestamp DESC
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
            SET status = ?
            WHERE id = ?
        """, (
            status,
            signal_id
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to update signal {signal_id}: {e}")

# ===============================
# Market Check Logic
# ===============================

def check_market_hit(symbol, timeframe, stop_loss, take_profit, direction):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=50)
    except Exception as e:
        logging.error(f"Market fetch failed for {symbol}: {e}")
        return None

    for candle in ohlcv:
        high = candle[2]
        low = candle[3]

        if direction.upper() == "LONG":
            if low <= stop_loss:
                return "STOP_LOSS"
            if high >= take_profit:
                return "TAKE_PROFIT"

        elif direction.upper() == "SHORT":
            if high >= stop_loss:
                return "STOP_LOSS"
            if low <= take_profit:
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
        signal_id, symbol, timeframe, entry, stop_loss, take_profit, direction = signal

        result = check_market_hit(symbol, timeframe, stop_loss, take_profit, direction)

        if result:
            logging.info(f"{symbol} {timeframe} hit {result}")

            update_signal_status(signal_id, result)

            message = (
                f"📊 **{symbol}** ({timeframe})\n"
                f"**Signal:** {direction}\n"
                f"**Result:** {result}\n"
                f"**Time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
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
