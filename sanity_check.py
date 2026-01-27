import sqlite3
import pandas as pd
import ccxt
import os
import sys
import logging
from datetime import datetime
from config import DB_FILE, ASSETS, TIMEFRAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def run_sanity_check():
    logging.info("--- STARTING NEXUS SANITY CHECK ---")
    errors = 0

    # 1. Check Configuration
    logging.info(f"Checking assets: {ASSETS}")
    if not ASSETS:
        logging.error("No ASSETS defined in config.py")
        errors += 1

    # 2. Check Database & Schema
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engine TEXT,
                asset TEXT, timeframe TEXT, signal TEXT, 
                entry REAL, sl REAL, tp REAL, confidence REAL, 
                reason TEXT, ts TEXT
            )
        """)
        
        # Check for 'engine' column (Migration check)
        cursor.execute("PRAGMA table_info(signals)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'engine' not in columns:
            logging.warning("Migration: Adding 'engine' column to signals table.")
            cursor.execute("ALTER TABLE signals ADD COLUMN engine TEXT DEFAULT 'core'")
        
        # Check latest record
        cursor.execute("SELECT ts FROM signals ORDER BY ts DESC LIMIT 1")
        last_ts = cursor.fetchone()
        if last_ts:
            logging.info(f"Latest signal in database: {last_ts[0]}")
        else:
            logging.info("Database is currently empty (New install).")
            
        conn.commit()
        conn.close()
        logging.info("Database check: PASSED")
    except Exception as e:
        logging.error(f"Database check: FAILED - {e}")
        errors += 1

    # 3. Check Exchange Connectivity
    try:
        ex = ccxt.binance()
        status = ex.fetch_status()
        logging.info(f"Exchange connectivity (Binance): {status['status']}")
    except Exception as e:
        logging.error(f"Exchange connectivity: FAILED - {e}")
        errors += 1

    # 4. Final Verdict
    if errors > 0:
        logging.error(f"Sanity Check failed with {errors} error(s).")
        sys.exit(1)
    else:
        logging.info("--- SANITY CHECK COMPLETE: SYSTEM HEALTHY ---")

if __name__ == "__main__":
    run_sanity_check()
