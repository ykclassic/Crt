import os
import ccxt
import pandas as pd
import numpy as np
import json
import requests
import time
from datetime import datetime, timezone

# --- CONFIGURATION ---
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "LINK/USDT", "DOGE/USDT", "TRX/USDT", "SUI/USDT", "PEPE/USDT"]
TIMEFRAMES = ["1h", "4h"]
EXCHANGE_NAME = "XT"
ATR_MULTIPLIER_STOP = 2.0
RR_RATIO = 1.5
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
STATE_FILE = "sent_alerts.json"

def get_exchange(name):
    ex = ccxt.xt({"enableRateLimit": True}) if name == "XT" else ccxt.gateio({"enableRateLimit": True})
    ex.load_markets()
    return ex

def compute_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))
    tr = pd.concat([(df['high']-df['low']), abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    # Supertrend logic
    hl2 = (df['high'] + df['low']) / 2
    df['ub'], df['lb'], df['uptrend'] = hl2 + (3.0 * df['atr']), hl2 - (3.0 * df['atr']), True
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] > df['ub'].iloc[i-1]: df.loc[df.index[i], 'uptrend'] = True
        elif df['close'].iloc[i-1] < df['lb'].iloc[i-1]: df.loc[df.index[i], 'uptrend'] = False
        else:
            df.loc[df.index[i], 'uptrend'] = df['uptrend'].iloc[i-1]
            if df['uptrend'].iloc[i] and df['lb'].iloc[i] < df['lb'].iloc[i-1]: df.loc[df.index[i], 'lb'] = df['lb'].iloc[i-1]
            elif not df['uptrend'].iloc[i] and df['ub'].iloc[i] > df['ub'].iloc[i-1]: df.loc[df.index[i], 'ub'] = df['ub'].iloc[i-1]
    df['st'] = np.where(df['uptrend'], df['lb'], df['ub'])
    return df

def get_signal(df):
    last = df.iloc[-1]
    sup_up = last["close"] > last["st"]
    sig = "LONG" if (last["close"] > last["ema20"] and last["rsi"] < 70 and sup_up) else \
          "SHORT" if (last["close"] < last["ema20"] and last["rsi"] > 30 and not sup_up) else "NEUTRAL"
    reg = "BULLISH" if last["close"] > last["ema50"] else "BEARISH"
    atr = last["atr"] if not np.isnan(last["atr"]) else last["close"] * 0.01
    entry = last["close"]
    stop = (entry - ATR_MULTIPLIER_STOP * atr) if sig == "LONG" else (entry + ATR_MULTIPLIER_STOP * atr)
    take = (entry + RR_RATIO * abs(entry - stop)) if sig == "LONG" else (entry - RR_RATIO * abs(entry - stop))
    return sig, reg, entry, stop, take

def run_alerts():
    ex = get_exchange(EXCHANGE_NAME)
    # Load state
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f: sent_state = json.load(f)
    else: sent_state = {}

    messages = []
    new_state = {}

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            try:
                data = ex.fetch_ohlcv(asset, tf, limit=100)
                df = compute_indicators(pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"]))
                sig, reg, entry, stop, take = get_signal(df)
                
                state_key = f"{asset}_{tf}"
                if sig != "NEUTRAL":
                    # Only alert if the signal is DIFFERENT from the last one we sent
                    if sent_state.get(state_key) != sig:
                        emoji = "🟢" if sig == "LONG" else "🔴"
                        messages.append(f"{emoji} **{sig}** {asset} ({tf})\nRegime: {reg}\nEntry: {entry:.4f}\nSL: {stop:.4f} | TP: {take:.4f}")
                    new_state[state_key] = sig
                else:
                    new_state[state_key] = "NEUTRAL"
                time.sleep(0.2)
            except: continue

    # Save state & Send alerts
    with open(STATE_FILE, "w") as f: json.dump(new_state, f)
    if messages and WEBHOOK_URL:
        requests.post(WEBHOOK_URL, json={"content": "\n".join(messages)})

if __name__ == "__main__":
    run_alerts()
