import sqlite3
import os
import sys
import logging
import ccxt
from datetime import datetime
from config import DB_FILE, ASSETS

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
        
        # Ensure table exists with all required columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                engine TEXT,
                asset TEXT, timeframe TEXT, signal TEXT, 
                entry REAL, sl REAL, tp REAL, confidence REAL, 
                reason TEXT, ts TEXT
            )
        """)
        
        # Column migration check
        cursor.execute("PRAGMA table_info(signals)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'engine' not in columns:
            logging.warning("Migration: Adding 'engine' column.")
            cursor.execute("ALTER TABLE signals ADD COLUMN engine TEXT DEFAULT 'core'")
            
        conn.commit()
        conn.close()
        logging.info("Database check: PASSED")
    except Exception as e:
        logging.error(f"Database check: FAILED - {e}")
        errors += 1

    # 3. Check Exchange Connectivity (REPLACED BINANCE WITH GATE.IO)
    # Using Gate.io because it has high uptime and fewer regional blocks for GitHub Actions
    try:
        logging.info("Testing connectivity to Gate.io...")
        ex = ccxt.gateio({'enableRateLimit': True})
        status = ex.fetch_status()
        logging.info(f"Exchange connectivity (Gate.io): {status.get('status', 'OK')}")
    except Exception as e:
        logging.error(f"Exchange connectivity: FAILED - {e}")
        # We check a secondary as a fallback
        try:
            logging.info("Testing fallback connectivity to XT.com...")
            xt = ccxt.xt()
            xt.fetch_status()
            logging.info("Fallback (XT): PASSED")
        except Exception as fallback_e:
            logging.error(f"All exchange connectivity failed: {fallback_e}")
            errors += 1

    # 4. Final Verdict
    if errors > 0:
        logging.error(f"Sanity Check failed with {errors} error(s).")
        sys.exit(1)
    else:
        logging.info("--- SANITY CHECK COMPLETE: SYSTEM HEALTHY ---")

if __name__ == "__main__":
    run_sanity_check()
