import sqlite3
import logging
from datetime import datetime, timedelta
from config import DB_FILE, DAYS_TO_KEEP

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def run_maintenance():
    logging.info(f"Maintenance start: {datetime.now().strftime('%Y-%m-%d')}")
    cutoff = (datetime.now() - timedelta(days=DAYS_TO_KEEP)).isoformat()

    if not os.path.exists(DB_FILE):
        logging.warning("DB not found")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM signals")
    before = cursor.fetchone()[0]
    
    cursor.execute("DELETE FROM signals WHERE ts < ?", (cutoff,))
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM signals")
    after = cursor.fetchone()[0]
    
    conn.close()
    logging.info(f"Removed {before - after} old records. Remaining: {after}")
    logging.info("Maintenance complete.")

if __name__ == "__main__":
    run_maintenance()
