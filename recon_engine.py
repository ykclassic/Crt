import sqlite3
import ccxt
import pandas as pd
import requests
import os
from datetime import datetime

# --- CONFIGURATION ---
DB_FILE = "nexus.db"
# Pulling from Environment Variable for Security
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

class ReconEngine:
    def __init__(self):
        # Initializing exchange nodes for real-time price auditing
        self.nodes = {
            'gate': ccxt.gateio(),
            'bitget': ccxt.bitget(),
            'xt': ccxt.xt()
        }

    def send_discord_alert(self, trade_data, outcome, current_p):
        """Sends the final Audit Result to Discord."""
        is_success = "SUCCESS" in outcome
        color = 0x3fb950 if is_success else 0xf85149
        emoji = "💰" if is_success else "🛡️"
        
        payload = {
            "username": "Nexus Recon Bot",
            "embeds": [{
                "title": f"{emoji} POST-TRADE AUDIT: {outcome}",
                "color": color,
                "fields": [
                    {"name": "Asset", "value": f"**{trade_data['asset']}**", "inline": True},
                    {"name": "Exchange", "value": trade_data.get('exchange', 'GATE').upper(), "inline": True},
                    {"name": "Entry Price", "value": f"{trade_data['entry']}", "inline": True},
                    {"name": "Exit Price", "value": f"**{current_p}**", "inline": True},
                    {"name": "Outcome", "value": "PROFIT" if is_success else "STOP LOSS", "inline": True},
                    {"name": "AI Confidence", "value": f"{trade_data['confidence']}%", "inline": True}
                ],
                "footer": {"text": f"Recon Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"}
            }]
        }
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=payload)

    def run_recon_cycle(self):
        if not os.path.exists(DB_FILE): return
        
        conn = sqlite3.connect(DB_FILE)
        # We only audit trades that haven't reached a final outcome yet
        df = pd.read_sql_query("SELECT * FROM signals WHERE status IS NULL OR status = 'ACTIVE'", conn)
        
        if df.empty:
            print("No active signals require reconciliation.")
            conn.close()
            return

        for _, trade in df.iterrows():
            exch = trade.get('exchange', 'gate').lower()
            node = self.nodes.get(exch, self.nodes['gate'])
            
            try:
                symbol = trade['asset'].replace("_", "/").upper()
                ticker = node.fetch_ticker(symbol)
                current_p = ticker['last']
                
                tp = float(trade['tp'])
                sl = float(trade['sl'])
                is_long = trade['signal'].upper() == 'LONG'
                
                outcome = None
                if is_long:
                    if current_p >= tp: outcome = "SUCCESS (TARGET HIT)"
                    elif current_p <= sl: outcome = "FAILED (STOP HIT)"
                else: # Short logic
                    if current_p <= tp: outcome = "SUCCESS (TARGET HIT)"
                    elif current_p >= sl: outcome = "FAILED (STOP HIT)"

                if outcome:
                    self.update_db_outcome(trade['id'], outcome)
                    self.send_discord_alert(trade, outcome, current_p)
                    print(f"Reconciled {symbol}: {outcome}")

            except Exception as e:
                print(f"Error auditing {trade['asset']} on {exch}: {e}")
        
        conn.close()

    def update_db_outcome(self, trade_id, outcome):
        status_label = 'SUCCESS' if 'SUCCESS' in outcome else 'FAILED'
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE signals SET status = ? WHERE id = ?", (status_label, trade_id))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    recon = ReconEngine()
    recon.run_recon_cycle()
