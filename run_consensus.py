import sqlite3
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from config import DB_FILE, WEBHOOK_URL, ENGINES

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def notify_discord(message):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": message})
        except Exception as e:
            logging.error(f"Consensus notify failed: {e}")

def get_latest_signals():
    conn = sqlite3.connect(DB_FILE)
    time_threshold = (datetime.now() - timedelta(hours=4)).isoformat()
    try:
        df = pd.read_sql_query("""
            SELECT asset, signal, confidence, reason, ts, engine 
            FROM signals 
            WHERE ts > ? 
            ORDER BY ts DESC
        """, conn, params=(time_threshold,))
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # Latest per asset+engine
    df = df.sort_values('ts').groupby(['asset', 'engine']).tail(1)
    return df

def run_consensus_check():
    df = get_latest_signals()
    if df.empty:
        logging.info("No recent signals for consensus")
        return

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

        side_emoji = "🔵" if direction == "LONG" else "🟠"
        
        if count >= 4:
            msg = (
                f"💎 **[URGENT: DIAMOND CONSENSUS]** 💎\n"
                f"**ALL {count} ENGINES AGREE ON {asset}**\n"
                f"{side_emoji} Signal: **{direction}**\n"
                f"📈 Avg Confidence: {avg_conf}%\n"
                f"**Technical Confluence:** {reasons}\n"
                f"⚠️ *Highest Tier Probability Setup*"
            )
            notify_discord(msg)
            logging.info(f"Diamond consensus: {asset} {direction}")
        elif count >= 3:
            msg = (
                f"🥇 **[GOLD CONSENSUS]**\n"
                f"**{count} Engines align on {asset}**\n"
                f"{side_emoji} Signal: **{direction}**\n"
                f"Confidence: {avg_conf}%\n"
                f"Basis: {reasons}"
            )
            notify_discord(msg)
            logging.info(f"Gold consensus: {asset} {direction}")

if __name__ == "__main__":
    run_consensus_check()
