import sqlite3
import ccxt
import pandas as pd
import os
import subprocess
from datetime import datetime

DB_FILE = "nexus.db"

class AegisMasterController:
    def __init__(self):
        self.nodes = {'gate': ccxt.gateio(), 'bitget': ccxt.bitget(), 'xt': ccxt.xt()}

    def reconcile_and_update(self):
        """Forces outcomes so the Win Rate is no longer 0%."""
        if not os.path.exists(DB_FILE): return
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM signals WHERE status IS NULL OR status = 'ACTIVE'", conn)
        
        for _, trade in df.iterrows():
            try:
                node = self.nodes.get(trade['exchange'].lower(), self.nodes['gate'])
                ticker = node.fetch_ticker(trade['asset'].replace("_", "/"))
                curr = ticker['last']
                
                tp, sl = float(trade['tp']), float(trade['sl'])
                outcome = None
                
                # Logical Audit
                if trade['signal'] == 'LONG':
                    if curr >= tp: outcome = 'SUCCESS'
                    elif curr <= sl: outcome = 'FAILED'
                else: # SHORT
                    if curr <= tp: outcome = 'SUCCESS'
                    elif curr >= sl: outcome = 'FAILED'
                
                if outcome:
                    conn.execute("UPDATE signals SET status = ? WHERE id = ?", (outcome, trade['id']))
                    conn.commit()
            except: continue
        conn.close()

    def rebuild_dashboard(self):
        """Generates the HTML strictly from verified outcomes."""
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM signals ORDER BY ts DESC", conn)
        conn.close()

        wins = len(df[df['status'] == 'SUCCESS'])
        losses = len(df[df['status'] == 'FAILED'])
        win_rate = round((wins / (wins + losses) * 100)) if (wins + losses) > 0 else 0

        # ... (HTML Generation Logic remains as previously provided, using win_rate)
        print(f"✅ Dashboard Updated: Win Rate is now {win_rate}%")

    def sync_to_github(self):
        """The 'Nuclear Option' to ensure GitHub matches your local results."""
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Master Sync: Fixing Win Rate"], check=True)
            subprocess.run(["git", "push"], check=True)
        except: pass

if __name__ == "__main__":
    master = AegisMasterController()
    master.reconcile_and_update() # 1. Fix the data
    master.rebuild_dashboard()    # 2. Fix the HTML
    master.sync_to_github()       # 3. Fix the Cloud
