import requests
import os
import pandas as pd
from datetime import datetime

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def forward_to_discord(filepath, title="Manual Export"):
    """
    Standard function used by the Dashboard buttons 
    to send any CSV to Discord instantly.
    """
    if not WEBHOOK_URL:
        print("Error: DISCORD_WEBHOOK_URL environment variable not set.")
        return False

    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return False
    
    try:
        with open(filepath, "rb") as f:
            payload = {"content": f"📤 **Nexus Dashboard Dispatch**: {title}"}
            files = {"file": (os.path.basename(filepath), f, "text/csv")}
            response = requests.post(WEBHOOK_URL, data=payload, files=files)
            
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"Failed to forward: {e}")
        return False

def generate_full_report(type_label="System Snapshot"):
    """Used for scheduled GitHub Action reports."""
    dbs = ["nexus_core.db", "hybrid_v1.db", "rangemaster.db", "nexus_ai.db"]
    frames = []
    for db in dbs:
        if os.path.exists(db):
            import sqlite3
            conn = sqlite3.connect(db)
            df = pd.read_sql_query("SELECT * FROM signals", conn)
            df['engine'] = db.replace(".db", "")
            frames.append(df)
            conn.close()
    
    if frames:
        master = pd.concat(frames)
        fname = f"Nexus_Full_Audit_{datetime.now().strftime('%Y%m%d')}.csv"
        master.to_csv(fname, index=False)
        forward_to_discord(fname, type_label)
        os.remove(fname)

if __name__ == "__main__":
    import sys
    # Handles automated reports from GitHub Actions
    if "--full" in sys.argv:
        generate_full_report("Automated Weekly/Monthly Audit")
