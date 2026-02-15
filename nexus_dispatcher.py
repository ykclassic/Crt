import requests
import logging
import os
import sqlite3
from datetime import datetime, timezone

# 1. Setup Paths - Targeting the existing root nexus_signals.db
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If this file is in a subfolder, we look one level up for the root DB
ROOT_DB_PATH = os.path.join(BASE_DIR, "nexus_signals.db")

if not os.path.exists(ROOT_DB_PATH):
    PARENT_DIR = os.path.dirname(BASE_DIR)
    ROOT_DB_PATH = os.path.join(PARENT_DIR, "nexus_signals.db")

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | DISPATCHER | %(levelname)s | %(message)s"
)

def initialize_database():
    """
    Ensures the database has the correct schema for AI learning.
    Added 'outcome' and 'pnl' to track performance.
    """
    try:
        conn = sqlite3.connect(ROOT_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dispatched_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                direction TEXT,
                tier TEXT,
                confidence REAL,
                entry REAL,
                stop_loss REAL,
                take_profit REAL,
                timestamp TEXT,
                status TEXT,
                outcome TEXT DEFAULT 'PENDING',
                pnl REAL DEFAULT 0.0
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")

def log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, status):
    """Saves the signal for both record-keeping and AI training."""
    try:
        conn = sqlite3.connect(ROOT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dispatched_alerts (
                pair, direction, tier, confidence, entry, 
                stop_loss, take_profit, timestamp, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair, direction, tier, confidence, entry, 
            stop_loss, take_profit, 
            datetime.now(timezone.utc).isoformat(),
            status
        ))
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id
    except Exception as e:
        logging.error(f"Failed to log signal to DB: {e}")
        return None

def update_signal_performance(signal_id, outcome, pnl):
    """
    Updates a signal with its actual market result.
    The AI uses this table as a 'Training Set' to improve accuracy.
    outcome: 'HIT_TP', 'HIT_SL', or 'MANUAL_CLOSE'
    """
    try:
        conn = sqlite3.connect(ROOT_DB_PATH)
        conn.execute("""
            UPDATE dispatched_alerts 
            SET outcome = ?, pnl = ?
            WHERE id = ?
        """, (outcome, pnl, signal_id))
        conn.commit()
        conn.close()
        logging.info(f"Signal {signal_id} performance updated: {outcome}")
    except Exception as e:
        logging.error(f"Failed to update performance for ID {signal_id}: {e}")

def dispatch_signal(pair, direction, tier,
                    confidence, entry,
                    stop_loss, take_profit):
    """
    Main entry point: Logs to DB, Dispatches to Discord.
    """
    initialize_database()

    # Create the message
    message = (
        f"🚀 **New Signal Dispatched**\n"
        f"**Asset:** {pair}\n"
        f"**Direction:** {direction}\n"
        f"**Tier:** {tier}\n"
        f"**Confidence:** {round(confidence, 2)}\n"
        f"**Entry:** {round(entry, 4)}\n"
        f"**Stop Loss:** {round(stop_loss, 4)}\n"
        f"**Take Profit:** {round(take_profit, 4)}"
    )

    # Attempt to send to Discord
    status = "INITIATED"
    if not WEBHOOK_URL:
        logging.warning(f"Webhook missing. Logging {pair} locally only.")
        status = "LOCAL_ONLY"
    else:
        try:
            response = requests.post(WEBHOOK_URL, json={"content": message}, timeout=10)
            if response.status_code in [200, 204]:
                status = "SUCCESS"
            else:
                status = f"HTTP_ERR_{response.status_code}"
        except Exception as e:
            status = f"NETWORK_ERR"
            logging.error(f"Discord POST failed: {e}")

    # Log to database and return the ID for future 'learning' updates
    signal_id = log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, status)
    
    if status == "SUCCESS":
        logging.info(f"Signal {pair} ({tier}) live on Discord. ID: {signal_id}")
    
    return signal_id
