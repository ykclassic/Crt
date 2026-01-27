import sqlite3
import logging
import os  # Added missing import
from datetime import datetime, timedelta
from config import DB_FILE, DAYS_TO_KEEP

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def run_maintenance():
    logging.info(f"Maintenance start: {datetime.now().strftime('%Y-%m-%d')}")
    
    if not os.path.exists(DB_FILE):
        logging.warning(f"Database file {DB_FILE} not found. Skipping maintenance.")
        return

    # Calculate the date limit for old records
    cutoff = (datetime.now() - timedelta(days=DAYS_TO_KEEP)).isoformat()

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 1. Count before cleanup
        cursor.execute("SELECT COUNT(*) FROM signals")
        before = cursor.fetchone()[0]
        
        # 2. Delete old records
        logging.info(f"Deleting records older than {cutoff}...")
        cursor.execute("DELETE FROM signals WHERE ts < ?", (cutoff,))
        
        # 3. VACUUM the database
        # This is essential to actually reduce the file size on disk for GitHub
        logging.info("Compressing database (VACUUM)...")
        cursor.execute("VACUUM")
        
        conn.commit()
        
        # 4. Count after cleanup
        cursor.execute("SELECT COUNT(*) FROM signals")
        after = cursor.fetchone()[0]
        
        conn.close()
        logging.info(f"Maintenance successful: Removed {before - after} records. Remaining: {after}")
        
    except Exception as e:
        logging.error(f"Maintenance failed: {e}")

if __name__ == "__main__":
    run_maintenance()
