# arbitrage_dashboard.py
# Streamlit Arbitrage Tracker (Fee-Adjusted + Alerts + Filters)

import streamlit as st
import ccxt
import pandas as pd
import time
import requests

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Arbitrage Tracker Pro", layout="wide")
st.title("📊 Multi-Exchange Arbitrage Tracker (Fee-Adjusted)")

# --------------------------------------------------
# TELEGRAM CONFIG (REPLACE WITH YOUR REAL VALUES)
# --------------------------------------------------
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

def send_telegram(message: str):
    if "YOUR_" in TELEGRAM_BOT_TOKEN:
        return  # Safety: prevents silent API errors
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, json=payload, timeout=5)

# --------------------------------------------------
# EXCHANGES (CCXT VERIFIED)
# --------------------------------------------------
ALL_EXCHANGES = {
    "Bitget": ccxt.bitget(),
    "Gate.io": ccxt.gateio(),
    "XT": ccxt.xt()
}

PAIRS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT"
]

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

selected_exchanges = st.sidebar.multiselect(
    "Select Exchanges",
    list(ALL_EXCHANGES.keys()),
    default=list(ALL_EXCHANGES.keys())
)

min_volume = st.sidebar.number_input(
    "Minimum 24h Quote Volume (USDT)",
    min_value=0.0,
    value=5_000_000.0,
    step=1_000_000.0
)

alert_threshold = st.sidebar.number_input(
    "Alert Spread % (after fees)",
    min_value=0.1,
    value=1.0,
    step=0.1
)

refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 5, 30, 10)

# --------------------------------------------------
# FEE TABLE (PUBLIC, CONSERVATIVE)
# --------------------------------------------------
TRADING_FEES = {
    "Bitget": 0.001,
    "Gate.io": 0.002,
    "XT": 0.002
}

# --------------------------------------------------
# CORE LOGIC
# --------------------------------------------------
def fetch_prices():
    results = []

    active_exchanges = {
        name: ALL_EXCHANGES[name]
        for name in selected_exchanges
    }

    for pair in PAIRS:
        price_map = {}
        volume_map = {}

        for name, ex in active_exchanges.items():
            try:
                ticker = ex.fetch_ticker(pair)
                last = ticker["last"]
                volume = ticker.get("quoteVolume", 0)

                if last and volume >= min_volume:
                    price_map[name] = last
                    volume_map[name] = volume
            except Exception:
                continue

        if len(price_map) < 2:
            continue

        buy_ex = min(price_map, key=price_map.get)
        sell_ex = max(price_map, key=price_map.get)

        buy_price = price_map[buy_ex]
        sell_price = price_map[sell_ex]

        fee_buy = buy_price * TRADING_FEES.get(buy_ex, 0)
        fee_sell = sell_price * TRADING_FEES.get(sell_ex, 0)

        net_profit = (sell_price - fee_sell) - (buy_price + fee_buy)
        net_pct = (net_profit / (buy_price + fee_buy)) * 100

        row = {
            "Pair": pair,
            "Buy @": buy_ex,
            "Buy Price": round(buy_price, 4),
            "Sell @": sell_ex,
            "Sell Price": round(sell_price, 4),
            "Net Profit": round(net_profit, 4),
            "Net %": round(net_pct, 2)
        }

        results.append(row)

        if net_pct >= alert_threshold:
            send_telegram(
                f"🚨 Arbitrage Alert\n{pair}\nBuy: {buy_ex}\nSell: {sell_ex}\nNet: {net_pct:.2f}%"
            )

    return pd.DataFrame(results)

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
placeholder = st.empty()

while True:
    df = fetch_prices()
    with placeholder.container():
        st.dataframe(
            df.sort_values("Net %", ascending=False),
            use_container_width=True
        )
        st.caption("Fee-adjusted arbitrage | REST polling | Public APIs")
    time.sleep(refresh_rate)
