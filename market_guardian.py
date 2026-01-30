import ccxt
import pandas as pd
import json
import os
import logging
import requests
from datetime import datetime
from config import (
    PERFORMANCE_FILE, BTC_CRASH_THRESHOLD, GLOBAL_VOLATILITY_CAP, 
    WEBHOOK_URL, ENGINES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def notify_guardian(msg):
    if WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": f"🛡️ **MARKET GUARDIAN**: {msg}"})

def run_guardian():
    logging.info("--- GUARDIAN REGIME CHECK INITIATED ---")
    ex = ccxt.gateio()
    
    try:
        # Monitor BTC as the Market Heartbeat
        data = ex.fetch_ohlcv("BTC/USDT", '1h', limit=24)
        df = pd.DataFrame(data, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        
        # 1. Calculate Price Change %
        last_close = df['c'].iloc[-1]
        prev_close = df['c'].iloc[-2]
        hourly_change = ((last_close - prev_close) / prev_close) * 100
        
        # 2. Calculate Volatility (ATR %)
        high_low = df['h'] - df['l']
        atr = high_low.rolling(14).mean().iloc[-1]
        atr_pct = atr / last_close

        # Load current performance file
        perf = {}
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                perf = json.load(f)

        # TRIGGER CHECK
        market_unsafe = False
        reason = ""

        if hourly_change <= BTC_CRASH_THRESHOLD:
            market_unsafe = True
            reason = f"Flash Crash Detected ({hourly_change:.2f}%)"
        
        if atr_pct >= GLOBAL_VOLATILITY_CAP:
            market_unsafe = True
            reason = f"Extreme Volatility ({atr_pct:.2%})"

        if market_unsafe:
            logging.warning(f"🚨 MARKET REGIME UNSAFE: {reason}. Triggering Global Stop.")
            # Move all engines to RECOVERY status
            for engine_id in ENGINES.keys():
                if engine_id not in perf: perf[engine_id] = {}
                perf[engine_id]["status"] = "RECOVERY"
                perf[engine_id]["reason"] = reason
            
            notify_guardian(f"🚨 **GLOBAL CIRCUIT BREAKER TRIGGERED**\nReason: `{reason}`\nAll engines moved to **RECOVERY** mode for capital protection.")
        else:
            logging.info("Market regime is stable.")

        # Save status
        with open(PERFORMANCE_FILE, "w") as f:
            json.dump(perf, f, indent=4)

    except Exception as e:
        logging.error(f"Guardian Error: {e}")

if __name__ == "__main__":
    run_guardian()
