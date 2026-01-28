import sqlite3
import pandas as pd
import requests
import logging
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def notify_diamond(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": f"💎 **DIAMOND CONSENSUS ALERT**\n{msg}"})

def run_consensus():
    logging.info("--- STARTING MULTI-TIMEFRAME CONSENSUS ---")
    conn = sqlite3.connect(DB_FILE)
    
    # Get fresh 1h signals
    entry_query = "SELECT * FROM signals WHERE timeframe = '1h' AND ts > datetime('now', '-2 hours')"
    df_entries = pd.read_sql_query(entry_query, conn)

    # Get latest Macro signals
    macro_query = "SELECT * FROM signals WHERE timeframe IN ('4h', '1d') ORDER BY ts DESC"
    df_macro = pd.read_sql_query(macro_query, conn)

    if df_entries.empty:
        logging.info("No fresh entries for consensus.")
        return

    for _, entry in df_entries.iterrows():
        asset = entry['asset']
        # Check for macro alignment
        asset_macro = df_macro[df_macro['asset'] == asset]
        
        if not asset_macro.empty:
            macro_signal = asset_macro.iloc[0]['signal']
            macro_tf = asset_macro.iloc[0]['timeframe']

            if entry['signal'] == macro_signal:
                # MATCH FOUND: 1h Entry aligns with 4h/1d Trend
                msg = (
                    f"━━━━━━━━━━━━━━━\n"
                    f"🌟 **Asset**: `{asset}`\n"
                    f"🔥 **Signal**: `{entry['signal']}`\n"
                    f"📍 **Entry Price**: `{entry['entry']:.4f}`\n"
                    f"🎯 **Target**: `{entry['tp']:.4f}`\n"
                    f"🛑 **Stop**: `{entry['sl']:.4f}`\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"✅ **1h Logic**: {entry['engine']} ({entry['reason']})\n"
                    f"✅ **{macro_tf} Trend**: Confirmed Bullish/Bearish Alignment"
                )
                notify_diamond(msg)
                logging.info(f"Diamond Match for {asset}")

    conn.close()

if __name__ == "__main__":
    run_consensus()
