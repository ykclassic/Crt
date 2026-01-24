import sqlite3
import pandas as pd
import ccxt
import json
import os
from datetime import datetime

# --- CONFIGURATION ---
DB_FILES = {
    "nexus_core": "nexus_core.db",
    "hybrid_v1": "hybrid_v1.db",
    "rangemaster": "rangemaster.db",
    "nexus_ai": "nexus_ai.db"
}
PERFORMANCE_FILE = "performance.json"
# Thresholds for the Kill Switch
KILL_THRESHOLD = 40.0    # Drop below this? Go to RECOVERY
RECOVERY_THRESHOLD = 50.0 # Climb above this? Go back to LIVE

def audit_all():
    # Initialize XT Exchange for price verification
    ex = ccxt.xt({"enableRateLimit": True})
    performance_report = {}

    # 1. Load existing performance data if it exists
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                performance_report = json.load(f)
        except:
            performance_report = {}

    for strat_id, db_path in DB_FILES.items():
        if not os.path.exists(db_path):
            continue
        
        conn = sqlite3.connect(db_path)
        try:
            # Load last 50 signals for auditing
            df = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 50", conn)
            
            if df.empty:
                continue

            wins = 0
            total_audited = 0
            
            # Fetch current prices for all assets in the DB to check status
            # In a production environment, you'd check historical OHLCV. 
            # For this suite, we use a simplified 'Last Price' check vs TP/SL.
            for _, row in df.iterrows():
                try:
                    ticker = ex.fetch_ticker(row['asset'])
                    current_price = ticker['last']
                    
                    # Logic: Did it hit TP or SL?
                    if row['signal'] == 'LONG':
                        if current_price >= row['tp']: wins += 1
                    else: # SHORT
                        if current_price <= row['tp']: wins += 1
                    
                    total_audited += 1
                except:
                    continue

            # Calculate Win Rate using LaTeX logic: 
            # $$WR = \frac{Wins}{Total} \times 100$$
            wr = round((wins / total_audited * 100), 2) if total_audited > 0 else 50.0
            
            # 2. Determine Status (Kill Switch / Recovery)
            current_status = performance_report.get(strat_id, {}).get("status", "LIVE")
            
            if current_status == "LIVE" and wr < KILL_THRESHOLD and total_audited > 5:
                new_status = "RECOVERY"
            elif current_status == "RECOVERY" and wr >= RECOVERY_THRESHOLD:
                new_status = "LIVE"
            else:
                new_status = current_status

            # 3. Update the report
            performance_report[strat_id] = {
                "win_rate": wr,
                "status": new_status,
                "sample_size": total_audited,
                "last_audit": datetime.now().isoformat()
            }

        finally:
            conn.close()

    # 4. Save the learning file
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance_report, f, indent=4)
    
    print(f"✅ Audit Sync Complete. {len(performance_report)} engines updated.")

if __name__ == "__main__":
    audit_all()
