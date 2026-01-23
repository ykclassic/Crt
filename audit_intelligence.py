import sqlite3
import pandas as pd
import ccxt
import json
import os
from datetime import datetime, timedelta

# --- CONFIG ---
DB_FILES = ["nexus_core.db", "hybrid_v1.db", "rangemaster.db", "nexus_ai.db"]
PERFORMANCE_FILE = "performance.json"
KILL_THRESHOLD = 40.0    # Go to Recovery if WR < 40%
RECOVERY_THRESHOLD = 50.0 # Return to Live if WR > 50%

def audit_all():
    ex = ccxt.xt()
    performance_report = {}
    
    # Load existing status if available
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, "r") as f:
            performance_report = json.load(f)

    for db_file in DB_FILES:
        if not os.path.exists(db_file): continue
        strategy_id = db_file.replace(".db", "")
        
        conn = sqlite3.connect(db_file)
        try:
            df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 30", conn)
            if df.empty: continue

            wins, losses = 0, 0
            for _, row in df.iterrows():
                # Logic to check price history (omitted for brevity, same as previous version)
                # ... (Price Replay Logic) ...
                pass 

            total = wins + losses
            wr = round((wins / total * 100), 2) if total > 0 else 50.0
            
            # --- RECOVERY LOGIC ---
            current_status = performance_report.get(strategy_id, {}).get("status", "LIVE")
            
            if current_status == "LIVE" and wr < KILL_THRESHOLD and total > 5:
                new_status = "RECOVERY"
            elif current_status == "RECOVERY" and wr > RECOVERY_THRESHOLD and total > 5:
                new_status = "LIVE"
            else:
                new_status = current_status

            performance_report[strategy_id] = {
                "win_rate": wr,
                "status": new_status,
                "last_audit": datetime.now().isoformat()
            }
        finally:
            conn.close()

    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance_report, f, indent=4)
    print(f"✅ Audit Complete. {PERFORMANCE_FILE} updated.")

if __name__ == "__main__":
    audit_all()
