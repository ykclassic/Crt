import sqlite3
import pandas as pd
import requests
import logging
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def notify_diamond(msg):
    if WEBHOOK_URL:
        requests.post(
            WEBHOOK_URL,
            json={"content": f"💎 **DIAMOND CONSENSUS ALERT**\n{msg}"},
            timeout=10
        )


def safe_format(value):
    """
    Safely format numeric values to 4 decimal places.
    Returns 'N/A' if value is None or invalid.
    """
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return "N/A"


def run_consensus():
    logging.info("--- STARTING MULTI-TIMEFRAME CONSENSUS ---")

    conn = sqlite3.connect(DB_FILE)

    # Fresh 1h entries (last 2 hours)
    entry_query = """
        SELECT *
        FROM signals
        WHERE timeframe = '1h'
        AND ts > datetime('now', '-2 hours')
    """
    df_entries = pd.read_sql_query(entry_query, conn)

    # Latest macro signals (4h / 1d)
    macro_query = """
        SELECT *
        FROM signals
        WHERE timeframe IN ('4h', '1d')
        ORDER BY ts DESC
    """
    df_macro = pd.read_sql_query(macro_query, conn)

    if df_entries.empty:
        logging.info("No fresh entries for consensus.")
        conn.close()
        return

    for _, entry in df_entries.iterrows():
        asset = entry.get('asset')

        if not asset:
            continue

        # Filter macro signals for same asset
        asset_macro = df_macro[df_macro['asset'] == asset]

        if asset_macro.empty:
            continue

        macro_signal = asset_macro.iloc[0].get('signal')
        macro_tf = asset_macro.iloc[0].get('timeframe')

        if entry.get('signal') == macro_signal:

            msg = (
                f"━━━━━━━━━━━━━━━\n"
                f"🌟 **Asset**: `{asset}`\n"
                f"🔥 **Signal**: `{entry.get('signal', 'N/A')}`\n"
                f"📍 **Entry Price**: `{safe_format(entry.get('entry'))}`\n"
                f"🎯 **Target**: `{safe_format(entry.get('tp'))}`\n"
                f"🛑 **Stop**: `{safe_format(entry.get('sl'))}`\n"
                f"━━━━━━━━━━━━━━━\n"
                f"✅ **1h Logic**: {entry.get('engine', 'N/A')} "
                f"({entry.get('reason', 'No reason provided')})\n"
                f"✅ **{macro_tf} Trend**: Confirmed Alignment"
            )

            notify_diamond(msg)
            logging.info(f"Diamond Match for {asset}")

    conn.close()


if __name__ == "__main__":
    run_consensus()
