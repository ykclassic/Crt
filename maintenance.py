import sqlite3
import os
import json
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FILES = ["nexus_core.db", "hybrid_v1.db", "rangemaster.db", "nexus_ai.db"]
DAYS_TO_KEEP = 30

def run_maintenance():
    print(f"🧹 Starting System Maintenance: {datetime.now().strftime('%Y-%m-%d')}")
    cutoff = (datetime.now() - timedelta(days=DAYS_TO_KEEP)).isoformat()

    for db in DB_FILES:
        if os.path.exists(db):
            try:
                conn = sqlite3.connect(db)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM signals")
                before = cursor.fetchone()[0]
                
                # Parameterized deletion
                cursor.execute("DELETE FROM signals WHERE ts < ?", (cutoff,))
                conn.commit()
                
                cursor.execute("SELECT COUNT(*) FROM signals")
                after = cursor.fetchone()[0]
                
                conn.close()
                print(f"✅ {db}: Removed {before - after} old records. Remaining: {after}")
            except Exception as e:
                print(f"❌ Error cleaning {db}: {e}")

    print("✨ Maintenance Complete. Databases are optimized.")
if __name__ == "__main__":
    run_maintenance()
