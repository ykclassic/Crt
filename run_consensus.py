import sqlite3
import pandas as pd
import requests
import logging
import os
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
    if not os.path.exists(DB_FILE):
        logging.warning("Database not found. Skipping consensus.")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_FILE)
    # Only look for signals from the last 4 hours for meaningful confluence
    time_threshold = (datetime.now() - timedelta(hours=4)).isoformat()
    
    try:
        df = pd.read_sql_query("""
            SELECT asset, signal, confidence, reason, ts, engine 
            FROM signals 
            WHERE ts > ? 
            ORDER BY ts DESC
        """, conn, params=(time_threshold,))
    except Exception as e:
        logging.error(f"Database read error in consensus: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # Get only the absolute latest signal per asset per engine
    df = df.sort_values('ts').groupby(['asset', 'engine']).tail(1)
    return df

def run_consensus_check():
    logging.info("Starting Nexus Consensus Analysis...")
    df = get_latest_signals()
    
    if df.empty:
        logging.info("No recent signals found for consensus analysis.")
        return

    # Group by asset and signal direction (LONG/SHORT)
    # We want to find where multiple engines agree on the same direction
    summary = df.groupby(['asset', 'signal']).agg({
        'engine': list,          # List of engine names
        'confidence': 'mean',    # Average confidence score
        'reason': lambda x: ' | '.join(set(x)) # Unique reasons combined
    }).reset_index()

    # Calculate count of agreeing engines
    summary['count'] = summary['engine'].apply(len)

    for _, row in summary.iterrows():
        count = row['count']
        asset = row['asset']
        direction = row['signal']
        avg_conf = round(row['confidence'], 2)
        reasons = row['reason']
        engine_names = ", ".join(row['engine'])

        side_emoji = "🔵" if direction == "LONG" else "🟠"
        
        # DIAMOND CONSENSUS (All currently active engines agree)
        # Note: If you have 2 engines active, count >= 2 is a full agreement.
        if count >= 3:
            msg = (
                f"💎 **[URGENT: DIAMOND CONSENSUS]** 💎\n"
                f"**STRATEGIC ALIGNMENT ON {asset}**\n"
                f"Engines: `{engine_names}`\n"
                f"{side_emoji} Signal: **{direction}**\n"
                f"📈 Avg Confidence: `{avg_conf}%`\n"
                f"**Technical Confluence:** {reasons}\n"
                f"⚠️ *Highest Tier Probability Setup*"
            )
            notify_discord(msg)
            logging.info(f"Diamond consensus: {asset} {direction} ({count} engines)")

        # GOLD CONSENSUS (Significant majority agreement)
        elif count == 2:
            msg = (
                f"🥇 **[GOLD CONSENSUS]**\n"
                f"**Engines `{engine_names}` align on {asset}**\n"
                f"{side_emoji} Signal: **{direction}**\n"
                f"Confidence: `{avg_conf}%`\n"
                f"Basis: {reasons}"
            )
            notify_discord(msg)
            logging.info(f"Gold consensus: {asset} {direction} (2 engines)")

    logging.info("Consensus check complete.")

if __name__ == "__main__":
    run_consensus_check()
