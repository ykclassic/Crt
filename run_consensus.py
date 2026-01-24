import sqlite3
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
DB_FILES = {
    "nexus_core": "nexus_core.db",
    "hybrid_v1": "hybrid_v1.db",
    "rangemaster": "rangemaster.db",
    "nexus_ai": "nexus_ai.db"
}
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
APP_NAME = "NEXUS CONSENSUS"

def notify_discord(message):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": message})

def get_latest_signals():
    all_signals = []
    # Lookback window: signals must have happened in the last 4 hours to be "current"
    time_threshold = (datetime.now() - timedelta(hours=4)).isoformat()

    for engine, db_path in DB_FILES.items():
        if not os.path.exists(db_path):
            continue
        
        try:
            conn = sqlite3.connect(db_path)
            # Fetch latest signals within the window
            query = f"SELECT asset, signal, confidence, reason, ts FROM signals WHERE ts > '{time_threshold}'"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if not df.empty:
                # Keep only the absolute latest signal per asset for this engine
                df = df.sort_values('ts').groupby('asset').tail(1)
                df['engine'] = engine
                all_signals.append(df)
        except Exception as e:
            print(f"Error reading {engine}: {e}")

    return pd.concat(all_signals) if all_signals else pd.DataFrame()

def run_consensus_check():
    df = get_latest_signals()
    if df.empty:
        print("No recent signals found for consensus.")
        return

    # Group by asset and signal (LONG/SHORT)
    # We want to see how many engines agree on the SAME direction for the SAME asset
    summary = df.groupby(['asset', 'signal']).agg({
        'engine': 'count',
        'confidence': 'mean',
        'reason': lambda x: ' | '.join(x.unique())
    }).reset_index()

    for _, row in summary.iterrows():
        count = row['engine']
        asset = row['asset']
        direction = row['signal']
        avg_conf = round(row['confidence'], 2)
        reasons = row['reason']

        # EMOJI LOGIC
        side_emoji = "🔵" if direction == "LONG" else "🟠"
        
        # --- DIAMOND CONSENSUS (4 Engines) ---
        if count >= 4:
            msg = (
                f"💎 **[URGENT: DIAMOND CONSENSUS]** 💎\n"
                f"**ALL 4 ENGINES AGREE ON {asset}**\n"
                f"{side_emoji} Signal: **{direction}**\n"
                f"📈 Avg Confidence: {avg_conf}%\n"
                f"--- \n"
                f"**Technical Confluence:** {reasons}\n"
                f"⚠️ *Highest Tier Probability Setup*"
            )
            notify_discord(msg)
            print(f"Diamond Match found for {asset}")

        # --- GOLD CONSENSUS (3 Engines) ---
        elif count == 3:
            msg = (
                f"🥇 **[GOLD CONSENSUS]**\n"
                f"**3 Engines align on {asset}**\n"
                f"{side_emoji} Signal: **{direction}**\n"
                f"Confidence: {avg_conf}%\n"
                f"Basis: {reasons}"
            )
            notify_discord(msg)
            print(f"Gold Match found for {asset}")

if __name__ == "__main__":
    run_consensus_check()
