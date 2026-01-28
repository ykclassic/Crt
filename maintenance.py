import sqlite3
import os
import shutil
import logging
import requests
from datetime import datetime, timedelta
from config import (
    DB_FILE, HISTORY_DB, MODEL_FILE, BACKUP_DIR, 
    PENDING_CLEANUP_HOURS, MAX_SIGNAL_AGE_DAYS, WEBHOOK_URL
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def notify_custodian(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": f"🧹 **CUSTODIAN**: {msg}"})

def run_maintenance():
    logging.info("--- PHASE 4: MAINTENANCE STARTING ---")
    
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    # 1. Backup the AI Brain
    if os.path.exists(MODEL_FILE):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(MODEL_FILE, f"{BACKUP_DIR}brain_v_{timestamp}.pkl")
        logging.info("AI Brain backed up.")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        # 2. Cleanup "Ghost Signals" (1h signals that never hit TP/SL after 24h)
        # These are likely invalid now and shouldn't be trained on
        cleanup_limit = (datetime.now() - timedelta(hours=PENDING_CLEANUP_HOURS)).isoformat()
        cursor.execute("DELETE FROM signals WHERE timeframe = '1h' AND ts < ?", (cleanup_limit,))
        ghosts_removed = cursor.rowcount
        
        # 3. Archive Old History
        # Ensure history table exists in history DB
        hist_conn = sqlite3.connect(HISTORY_DB)
        hist_conn.execute("CREATE TABLE IF NOT EXISTS signals_archive AS SELECT * FROM signals WHERE 1=0")
        
        archive_limit = (datetime.now() - timedelta(days=MAX_SIGNAL_AGE_DAYS)).isoformat()
        
        # Copy to Archive
        cursor.execute(f"ATTACH DATABASE '{HISTORY_DB}' AS hist")
        cursor.execute("INSERT INTO hist.signals_archive SELECT * FROM main.signals WHERE ts < ?", (archive_limit,))
        cursor.execute("DELETE FROM main.signals WHERE ts < ?", (archive_limit,))
        archived_count = cursor.rowcount
        
        # 4. Vacuum the DB
        cursor.execute("VACUUM")
        
        conn.commit()
        hist_conn.close()
        
        msg = f"Maintenance Complete.\n• Ghosts Purged: `{ghosts_removed}`\n• Archived to History: `{archived_count}`\n• Database Vacuumed & Brain Backed up."
        notify_custodian(msg)
        logging.info(msg)

    except Exception as e:
        logging.error(f"Maintenance Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_maintenance()
