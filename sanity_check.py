import sqlite3
import os
import sys
import logging
import ccxt
from config import DB_FILE, ASSETS

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def run_sanity_check():
    logging.info("--- STARTING NEXUS SANITY CHECK ---")
    errors = 0

    # 1. Check Database
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.close()
        logging.info("Database check: PASSED")
    except Exception as e:
        logging.error(f"Database check: FAILED - {e}")
        errors += 1

    # 2. Check Exchange Connectivity (Gate.io)
    try:
        logging.info("Testing Data Fetch from Gate.io...")
        ex = ccxt.gateio({'enableRateLimit': True})
        # Fetching 1 candle of BTC/USDT to prove we can reach the API
        ex.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=1)
        logging.info("Exchange connectivity (Gate.io): PASSED")
    except Exception as e:
        logging.error(f"Gate.io connectivity: FAILED - {e}")
        
        # Fallback to XT
        try:
            logging.info("Testing fallback connectivity to XT.com...")
            xt = ccxt.xt()
            xt.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=1)
            logging.info("Fallback (XT): PASSED")
        except Exception as fallback_e:
            logging.error(f"All exchange connectivity failed: {fallback_e}")
            errors += 1

    if errors > 0:
        logging.error(f"Sanity Check failed with {errors} error(s).")
        sys.exit(1)
    else:
        logging.info("--- SANITY CHECK COMPLETE: SYSTEM HEALTHY ---")

if __name__ == "__main__":
    run_sanity_check()
