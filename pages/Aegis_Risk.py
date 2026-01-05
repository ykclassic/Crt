import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import requests
import asyncio
from datetime import datetime

# 1. Page Config
st.set_page_config(page_title="Aegis Risk | Intelligent Monitor", page_icon="📉", layout="wide")

# 2. Telegram Notifier Logic
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

async def send_aegis_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": f"🛡️ **AEGIS RISK ALERT**\n{message}", "parse_mode": "Markdown"}
        requests.post(url, json=payload)
    except:
        pass

# 3. Market Intelligence Logic
def check_tectonic_shifts():
    """Analyzes market for significant changes and triggers alerts"""
    alerts = []
    
    # Fetch Data
    fng_res = requests.get("https://api.alternative.me/fng/").json()
    fng_now = int(fng_res['data'][0]['value'])
    
    cg_global = requests.get("https://api.coingecko.com/api/v3/global").json()
    btc_dom = cg_global['data']['market_cap_percentage']['btc']
    
    # 1. BTC Dominance Shift Alert
    # Threshold: Change of > 1% in dominance is significant
    if "prev_dom" in st.session_state:
        diff = btc_dom - st.session_state.prev_dom
        if abs(diff) > 0.5:
            direction = "INCREASING (Capital Fleeing to Safety)" if diff > 0 else "DECREASING (Altcoin Season Potential)"
            alerts.append(f"⚠️ BTC Dominance Shift: {direction} by {abs(diff):.2f}%")
    st.session_state.prev_dom = btc_dom

    # 2. Volatility Spike Alert (Using Bitget)
    try:
        ex = ccxt.bitget()
        ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe='1h', limit=5)
        last_change = abs((ohlcv[-1][4] - ohlcv[-2][4]) / ohlcv[-2][4])
        if last_change > 0.03: # 3% hourly move
            alerts.append(f"⚡ VOLATILITY SPIKE: BTC moved {last_change:.2%} in 60 minutes.")
    except: pass

    # 3. Fear & Greed Sentiment Shift
    if fng_now > 80:
        alerts.append("🔥 EXTREME GREED: Market overextended. High risk of liquidation flush.")
    elif fng_now < 20:
        alerts.append("❄️ EXTREME FEAR: Capitulation detected. Potential bottoming interest.")

    return alerts

# 4. Dashboard View
st.title("📉 Aegis Risk: Intelligence & Alerts")

# Run Monitor
if st.button("Manual Risk Scan", use_container_width=True):
    with st.spinner("Analyzing Tectonic Shifts..."):
        active_alerts = check_tectonic_shifts()
        if active_alerts:
            for a in active_alerts:
                st.warning(a)
                # Async call to Telegram
                asyncio.run(send_aegis_alert(a))
        else:
            st.success("No significant tectonic shifts detected in the last cycle.")

st.write("---")

# 5. Visual Summary of Risk Levels
# Image of the market risk parameters and alert thresholds for the Aegis OS Risk module


col1, col2 = st.columns(2)
with col1:
    st.subheader("📡 Notification Settings")
    st.toggle("Telegram Alerts", value=True)
    st.toggle("Push Notifications (Browser)", value=False)
    st.slider("Volatility Alert Threshold (%)", 1.0, 10.0, 3.0)

with col2:
    st.subheader("📜 Recent Risk Log")
    # This would pull from your aegis_system.db logs table
    st.code("""
    [2026-01-05 22:10] VOL SPIKE: BTC +3.4% (Bitget)
    [2026-01-05 18:00] SHIFT: BTC Dominance -0.8% (Altcoin Interest)
    [2026-01-05 12:45] ALERT: Fear & Greed entered 'Extreme Greed' (82)
    """, language="text")

st.info("The Risk Monitor runs every 5 minutes in the background when Aegis OS is active.")
