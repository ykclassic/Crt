import sqlite3
import ccxt
import pandas as pd
import requests
import os
import subprocess
from datetime import datetime

# --- CONFIGURATION ---
DB_FILE = "nexus.db"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")
TEST_START_BALANCE = 100.0  
FIXED_RISK = 5.0            

class ReconEngine:
    def __init__(self):
        self.nodes = {
            'gate': ccxt.gateio(),
            'bitget': ccxt.bitget(),
            'xt': ccxt.xt()
        }

    def force_github_sync(self):
        """Pushes the local database to GitHub to ensure dashboard matches the app."""
        try:
            subprocess.run(["git", "add", DB_FILE], check=True)
            subprocess.run(["git", "commit", "-m", "🔄 Sync: Aligning Dashboard with App Alerts"], check=True)
            subprocess.run(["git", "push"], check=True)
            print("🚀 GitHub Sync Complete: Dashboard should now match App.")
        except Exception as e:
            print(f"⚠️ Sync Error (Likely no git configured locally): {e}")

    def send_discord_alert(self, trade_data, outcome, current_p):
        is_success = "SUCCESS" in outcome
        color = 0x3fb950 if is_success else 0xf85149
        payload = {
            "username": "Nexus Recon Bot",
            "embeds": [{
                "title": f"🛡️ POST-TRADE AUDIT: {outcome}",
                "color": color,
                "fields": [
                    {"name": "Asset", "value": trade_data['asset'], "inline": True},
                    {"name": "Outcome", "value": "PROFIT" if is_success else "LOSS", "inline": True},
                    {"name": "Final Price", "value": str(current_p), "inline": True},
                ],
                "footer": {"text": f"Audit Time: {datetime.now().strftime('%H:%M:%S UTC')}"}
            }]
        }
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=payload)

    def run_recon_cycle(self):
        if not os.path.exists(DB_FILE): return
        conn = sqlite3.connect(DB_FILE)
        # Prioritize trades the dashboard says are 'SCANNING'
        df = pd.read_sql_query("SELECT * FROM signals WHERE status IS NULL OR status = 'ACTIVE'", conn)
        
        updated_any = False
        for _, trade in df.iterrows():
            exch = trade.get('exchange', 'gate').lower()
            node = self.nodes.get(exch, self.nodes['gate'])
            try:
                symbol = trade['asset'].replace("_", "/").upper()
                ticker = node.fetch_ticker(symbol)
                current_p = ticker['last']
                
                tp, sl = float(trade['tp']), float(trade['sl'])
                is_long = trade['signal'].upper() == 'LONG'
                
                outcome = None
                if is_long:
                    if current_p >= tp: outcome = "SUCCESS"
                    elif current_p <= sl: outcome = "FAILED"
                else:
                    if current_p <= tp: outcome = "SUCCESS"
                    elif current_p >= sl: outcome = "FAILED"

                if outcome:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE signals SET status = ? WHERE id = ?", (outcome, trade['id']))
                    conn.commit()
                    self.send_discord_alert(trade, outcome, current_p)
                    updated_any = True
            except Exception as e:
                print(f"Audit Error: {e}")
        
        conn.close()
        if updated_any:
            self.force_github_sync()

if __name__ == "__main__":
    recon = ReconEngine()
    recon.run_recon_cycle()
