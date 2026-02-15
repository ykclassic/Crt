import requests
import logging
import os
import sqlite3
from datetime import datetime, timezone

# Locate the existing nexus_signals.db in your root folder
# This looks for the DB in the current directory or the parent directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DB_PATH = os.path.join(BASE_DIR, "nexus_signals.db")

# Fallback: If not in current dir, check one level up (common if script is in /src/)
if not os.path.exists(ROOT_DB_PATH):
    PARENT_DIR = os.path.dirname(BASE_DIR)
    ROOT_DB_PATH = os.path.join(PARENT_DIR, "nexus_signals.db")

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | DISPATCH | %(levelname)s | %(message)s"
)

def ensure_table_exists():
    """Ensures the dispatched_alerts table exists in the existing root DB."""
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
                status TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error accessing existing DB at {ROOT_DB_PATH}: {e}")

def log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, status):
    """Saves the dispatch attempt to the existing nexus_signals.db."""
    try:
        conn = sqlite3.connect(ROOT_DB_PATH)
        conn.execute("""
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
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to write to {ROOT_DB_PATH}: {e}")

def dispatch_signal(pair, direction, tier,
                    confidence, entry,
                    stop_loss, take_profit):
    
    # Ensure the table is ready in the existing file
    ensure_table_exists()

    if not WEBHOOK_URL:
        logging.warning("Webhook not configured")
        log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, "MISSING_WEBHOOK")
        return

    message = f"""
Asset: {pair}
Direction: {direction}
Tier: {tier}
Confidence: {round(confidence, 2)}
Entry: {round(entry, 4)}
Stop Loss: {round(stop_loss, 4)}
Take Profit: {round(take_profit, 4)}
"""

    try:
        response = requests.post(WEBHOOK_URL, json={"content": message})
        
        # 204 No Content is the standard Discord success response
        if response.status_code in [200, 204]:
            logging.info(f"{pair} dispatched to Discord")
            log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, "SUCCESS")
        else:
            status_msg = f"HTTP_{response.status_code}"
            logging.error(f"{pair} Discord error: {status_msg}")
            log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, status_msg)
            
    except Exception as e:
        logging.error(f"{pair} dispatch failed: {e}")
        log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, f"ERROR: {str(e)}")
