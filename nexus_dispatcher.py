import requests
import logging
import os
import sqlite3
from datetime import datetime, timezone
from config import DB_FILE, WEBHOOK_URL

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DB_PATH = os.path.join(BASE_DIR, DB_FILE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DISPATCHER | %(levelname)s | %(message)s")

def initialize_dispatch_table():
    try:
        conn = sqlite3.connect(ROOT_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dispatched_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                direction TEXT,
                tier TEXT,
                confidence REAL,
                entry REAL,
                stop_loss REAL,
                take_profit REAL,
                timestamp TEXT,
                status TEXT,
                outcome TEXT DEFAULT 'PENDING',
                pnl REAL DEFAULT 0.0
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")

def log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, status):
    try:
        conn = sqlite3.connect(ROOT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dispatched_alerts (
                pair, direction, tier, confidence, entry, 
                stop_loss, take_profit, timestamp, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair, direction, tier, confidence, entry, 
            stop_loss, take_profit, 
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            status
        ))
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id
    except Exception as e:
        logging.error(f"Failed to log signal to DB: {e}")
        return None

def update_signal_performance(signal_id, outcome, pnl):
    """Updates the signal outcome and PnL in the dispatched_alerts table for backtesting."""
    try:
        conn = sqlite3.connect(ROOT_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE dispatched_alerts
            SET outcome = ?, pnl = ?
            WHERE id = ?
        """, (outcome, pnl, signal_id))
        
        conn.commit()
        conn.close()
        logging.info(f"Successfully updated signal {signal_id} outcome to {outcome} (PnL: {pnl}%)")
    except Exception as e:
        logging.error(f"Failed to update signal performance for ID {signal_id}: {e}")

def dispatch_signal(pair, direction, tier, confidence, entry, stop_loss, take_profit):
    initialize_dispatch_table()

    # Discord Embed Visual Formatting Mapping
    trade_type = "BUY" if direction == "LONG" else "SELL"
    color = 0x00FF00 if direction == "LONG" else 0xFF0000
    setup_type = "bullish" if direction == "LONG" else "bearish"
    clean_pair = pair.replace("/", "")
    
    # Calculate staggered Take Profit targets (40/30/30 distribution)
    tp3 = take_profit
    tp1 = entry + (tp3 - entry) * 0.33
    tp2 = entry + (tp3 - entry) * 0.66
    
    # Risk/Reward Ratio Calculation
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    rr_ratio = reward / risk if risk > 0 else 0

    # Ensure localized formatting for prices
    description_text = (
        f"{tier} Consensus indicates a strong {setup_type} setup on the 1-hour timeframe.\n\n"
        f"**Entry Zone:** ${entry * 0.998:,.4f} - ${entry * 1.002:,.4f}\n"
        f"**Take Profit 1:** ${tp1:,.4f} (40% of position)\n"
        f"**Take Profit 2:** ${tp2:,.4f} (30% of position)\n"
        f"**Take Profit 3:** ${tp3:,.4f} (30% of position)\n"
        f"**Stop Loss:** ${stop_loss:,.4f}\n"
        f"**Risk/Reward Ratio:** 1:{rr_ratio:.1f}\n"
        f"**Timeframe:** 1H\n"
        f"**Exchange:** XT.com\n\n"
        f"Always manage your risk. Not financial advice."
    )

    payload = {
        "username": "Signal Bot",
        "embeds": [{
            "title": f"🚨 NEW TRADING SIGNAL: {trade_type} ${clean_pair} (Spot/Futures)",
            "description": description_text,
            "color": color
        }]
    }

    status = "INITIATED"
    if not WEBHOOK_URL:
        logging.warning(f"Webhook missing. Logging {pair} locally only.")
        status = "LOCAL_ONLY"
    else:
        try:
            response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
            if response.status_code in [200, 204]:
                status = "SUCCESS"
            else:
                status = f"HTTP_ERR_{response.status_code}"
        except Exception as e:
            status = "NETWORK_ERR"
            logging.error(f"Discord POST failed: {e}")

    signal_id = log_to_database(pair, direction, tier, confidence, entry, stop_loss, take_profit, status)
    
    if status == "SUCCESS":
        logging.info(f"Signal {pair} ({tier}) live on Discord. ID: {signal_id}")

    return signal_id
