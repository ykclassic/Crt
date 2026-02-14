import sqlite3
import ccxt
import requests
import logging
import time
import os
from datetime import datetime, timezone
from config import DB_FILE

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ALERT_MONITOR | %(levelname)s | %(message)s"
)

# ===============================
# Configuration
# ===============================

POLL_INTERVAL = 300          # seconds (5 minutes)
MAX_SIGNALS_PER_CYCLE = 50   # CI safety cap
SINGLE_RUN = os.getenv("SINGLE_RUN", "false").lower() == "true"

# ===============================
# Exchange Setup
# ===============================

exchange = ccxt.gateio({
    "enableRateLimit": True,
    "timeout": 15000
})

# ===============================
# Database Helpers
# ===============================

def fetch_active_signals():
    conn = sqlite3.connect(DB_FILE)
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


def update_signal_status(signal_id, status):
    conn = sqlite3.connect(DB_FILE)
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


# ===============================
# Market Check Logic
# ===============================

def check_market_hit(asset, timeframe, sl, tp, signal_type):
    try:
        ohlcv = exchange.fetch_ohlcv(asset, timeframe, limit=50)

        for candle in ohlcv:
            high, low = candle[2], candle[3]

            if signal_type == "LONG":
                if high >= tp:
                    return "TP"
                if low <= sl:
                    return "SL"

            if signal_type == "SHORT":
                if low <= tp:
                    return "TP"
                if high >= sl:
                    return "SL"

        return None

    except Exception as e:
        logging.error(f"Market check failed for {asset}: {e}")
        return None


# ===============================
# Alert Sender
# ===============================

def send_alert(signal_id, asset, direction, result, price):
    if not WEBHOOK_URL:
        logging.error("WEBHOOK_URL not configured")
        return

    message = (
        f"🎯 SIGNAL RESOLVED\n"
        f"━━━━━━━━━━━━━━━\n"
        f"ID: {signal_id}\n"
        f"Asset: {asset}\n"
        f"Direction: {direction}\n"
        f"Outcome: {result}\n"
        f"Price Hit: {round(price, 4)}\n"
        f"Time: {datetime.now(timezone.utc).isoformat()}"
    )

    try:
        response = requests.post(
            WEBHOOK_URL,
            json={"content": message},
            timeout=10
        )

        if response.status_code not in [200, 204]:
            logging.error(f"Webhook error: {response.status_code} {response.text}")
        else:
            logging.info(f"{asset} {result} alert sent")

    except Exception as e:
        logging.error(f"Dispatch failed: {e}")


# ===============================
# Monitor Loop
# ===============================

def monitor_cycle():
    signals = fetch_active_signals()

    if not signals:
        logging.info("No active signals")
        return

    for signal in signals:
        signal_id, asset, timeframe, entry, sl, tp, direction = signal

        result = check_market_hit(asset, timeframe, sl, tp, direction)

        if result:
            hit_price = tp if result == "TP" else sl
            send_alert(signal_id, asset, direction, result, hit_price)
            update_signal_status(signal_id, result)


def run_monitor():
    logging.info("Starting continuous alert monitor")

    while True:
        monitor_cycle()

        if SINGLE_RUN:
            logging.info("Single run mode enabled. Exiting.")
            break

        logging.info(f"Sleeping {POLL_INTERVAL} seconds...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_monitor()
