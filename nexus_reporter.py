import requests
import os
import pandas as pd
from datetime import datetime

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def forward_to_discord(filepath, title="Custom Report"):
    """Forwards ANY file to Discord instantly."""
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
    
    with open(filepath, "rb") as f:
        payload = {"content": f"📤 **Nexus Dispatch**: {title}"}
        files = {"file": (os.path.basename(filepath), f, "text/csv")}
        requests.post(WEBHOOK_URL, data=payload, files=files)
    print(f"Sent {filepath} to Discord.")

def generate_full_report(type_label="System Snapshot"):
    """Gathers all signals into one master CSV and sends it."""
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
        fname = f"Nexus_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        master.to_csv(fname, index=False)
        forward_to_discord(fname, type_label)
        os.remove(fname)

if __name__ == "__main__":
    import sys
    # Usage: python nexus_reporter.py --full or --forward path/to/file.csv
    if "--full" in sys.argv:
        generate_full_report("Automated System Audit")
    elif "--forward" in sys.argv and len(sys.argv) > 2:
        forward_to_discord(sys.argv[2], "Manual Dashboard Export")
