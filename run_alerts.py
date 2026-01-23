import os
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
import time
import json
from datetime import datetime

# --- CONFIGURATION ---
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT", "DOGE/USDT", "TRX/USDT", "SUI/USDT", "PEPE/USDT"]
TIMEFRAMES = ["1h", "4h"]
EXCHANGE_NAME = "XT" 
ATR_MULTIPLIER_STOP = 2.0
RR_RATIO = 1.5
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DB_FILE = "nexus_core.db"
PERFORMANCE_FILE = "performance.json"
APP_NAME = "NEXUS CORE"

def notify(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})

def get_exchange(name):
    name = name.upper()
    if name == "XT": return ccxt.xt({"enableRateLimit": True})
    elif name == "GATE": return ccxt.gateio({"enableRateLimit": True})
    elif name == "BITGET": return ccxt.bitget({"enableRateLimit": True})
    else: raise ValueError(f"Unknown exchange: {name}")

def get_learned_confidence():
    """Reads the audit file to get the real win rate."""
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                data = json.load(f)
                # Lookup performance for 'nexus_core'
                stats = data.get("nexus_core", {"win_rate": 50.0})
                return stats["win_rate"]
    except:
        pass
    return 50.0 # Default if no data exists

def compute_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    tr = pd.concat([(df['high']-df['low']), abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df

def run_alerts():
    ex = get_exchange(EXCHANGE_NAME)
    ex.load_markets()

    # Initialize Database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT, timeframe TEXT, signal TEXT, 
            entry REAL, sl REAL, tp REAL, confidence REAL, ts TEXT
        )
    """)
    conn.commit()

    # Get the 'Learned' confidence from history
    current_win_rate = get_learned_confidence()

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            try:
                data = ex.fetch_ohlcv(asset, tf, limit=100)
                df = compute_indicators(pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"]))

                last = df.iloc[-1]
                entry = last["close"]
                atr = last["atr"] if not np.isnan(last["atr"]) else entry * 0.01

                signal = "NEUTRAL"
                if last["close"] > last["ema20"] and last["rsi"] < 70:
                    signal = "LONG"
                elif last["close"] < last["ema20"] and last["rsi"] > 30:
                    signal = "SHORT"

                if signal != "NEUTRAL":
                    cursor.execute("SELECT signal FROM signals WHERE asset=? AND timeframe=? ORDER BY id DESC LIMIT 1", (asset, tf))
                    last_saved = cursor.fetchone()

                    if not last_saved or last_saved[0] != signal:
                        sl = (entry - ATR_MULTIPLIER_STOP * atr) if signal == "LONG" else (entry + ATR_MULTIPLIER_STOP * atr)
                        tp = (entry + RR_RATIO * abs(entry - sl)) if signal == "LONG" else (entry - RR_RATIO * abs(entry - sl))
                        
                        # Apply learning: The confidence is now the historical Win Rate
                        confidence = current_win_rate

                        cursor.execute("""
                            INSERT INTO signals (asset, timeframe, signal, entry, sl, tp, confidence, ts)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (asset, tf, signal, entry, sl, tp, confidence, datetime.now().isoformat()))
                        conn.commit()

                        # Discord Alert
                        emoji = "🟢" if signal == "LONG" else "🔴"
                        status_msg = "🔥 OVERPERFORMING" if confidence > 60 else "⚠️ UNSTABLE" if confidence < 45 else "📊 STABLE"
                        
                        notify(
                            f"{emoji} **{signal}** {asset} ({tf})\n"
                            f"**Confidence: {confidence}%** ({status_msg})\n"
                            f"---\n"
                            f"Entry: {entry:.4f}\n"
                            f"SL: {sl:.4f} | TP: {tp:.4f}"
                        )

                time.sleep(0.1)
            except Exception as e:
                print(f"Error processing {asset}: {e}")

    conn.close()

if __name__ == "__main__":
    run_alerts()
