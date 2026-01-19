import sqlite3
import os
import requests
from datetime import datetime

# --- CONFIG ---
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS CONSENSUS"
DATABASES = ["nexus_core.db", "hybrid_v1.db", "rangemaster.db", "nexus_ai.db"]
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

def notify(msg):
    if WEBHOOK:
        requests.post(WEBHOOK, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def get_latest_signal(db_path, asset):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT signal, entry, sl, tp FROM signals WHERE asset=? ORDER BY id DESC LIMIT 1", (asset,))
        row = cursor.fetchone()
        conn.close()
        return row # (signal, entry, sl, tp)
    except:
        return None

def run_consensus():
    for asset in ASSETS:
        signals = []
        entries = []
        
        # Collect opinions from all brains
        for db in DATABASES:
            data = get_latest_signal(db, asset)
            if data:
                signals.append(data[0]) # The signal (LONG/SHORT)
                entries.append(data[1]) # The price
        
        if not signals: continue
        
        # Count votes
        long_votes = signals.count("LONG")
        short_votes = signals.count("SHORT")
        total_engines = len(DATABASES)
        
        # Logic: Require at least 2 engines to agree
        consensus_found = False
        final_sig = ""
        
        if long_votes >= 2:
            consensus_found = True
            final_sig = "LONG"
        elif short_votes >= 2:
            consensus_found = True
            final_sig = "SHORT"
            
        if consensus_found:
            avg_entry = sum(entries) / len(entries)
            vote_count = long_votes if final_sig == "LONG" else short_votes
            
            # Use a unique emoji for high-confidence consensus
            emoji = "💎" if vote_count >= 3 else "🤝"
            
            msg = (
                f"{emoji} **CONSENSUS DETECTED**\n"
                f"**Asset:** {asset}\n"
                f"**Direction:** {final_sig}\n"
                f"**Agreement:** {vote_count}/{total_engines} Engines\n"
                f"**Avg Entry:** {avg_entry:.2f}"
            )
            notify(msg)

if __name__ == "__main__":
    run_consensus()
