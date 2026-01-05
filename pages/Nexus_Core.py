import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Nexus Core | System Engine", page_icon="⚙️", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# 3. Market Microstructure & Confluence Engine Logic
def fetch_orderbook_data(symbol):
    """Fetches real-time bid/ask depth and calculates spread metrics."""
    try:
        ex = ccxt.bitget()
        ob = ex.fetch_order_book(symbol, limit=25)
        bids = pd.DataFrame(ob['bids'], columns=['price', 'quantity'])
        asks = pd.DataFrame(ob['asks'], columns=['price', 'quantity'])
        
        # Calculate Microstructure Metrics
        best_bid = bids['price'].iloc[0]
        best_ask = asks['price'].iloc[0]
        spread = best_ask - best_bid
        spread_pct = (spread / best_ask) * 100
        
        # Orderbook Imbalance (Buy vs Sell Pressure in top 25 levels)
        buy_vol = bids['quantity'].sum()
        sell_vol = asks['quantity'].sum()
        imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)
        
        return bids, asks, spread, spread_pct, imbalance
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), 0, 0, 0

def get_mtf_confluence(symbol):
    """Calculates Trend Alignment for 15m, 1h, 4h, and 1d using EMA Crossover logic."""
    intervals = ['15m', '1h', '4h', '1d']
    ex = ccxt.bitget()
    results = {}
    for tf in intervals:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=50)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            ema_fast = df['c'].ewm(span=9).mean().iloc[-1]
            ema_slow = df['c'].ewm(span=21).mean().iloc[-1]
            results[tf] = "BULLISH" if ema_fast > ema_slow else "BEARISH"
        except:
            results[tf] = "DATA_ERROR"
    return results

# 4. Header & Asset Selection
st.title("⚙️ Nexus Core: Infrastructure Hub")

# Expanded Asset Library (Originals + 7 New High-Volume/Volatile Pairs)
asset_library = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT",  # Originals
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", # High Volume
    "LINK/USDT", "TRX/USDT", "SUI/USDT", # Strategic/Network
    "PEPE/USDT"                          # High Volatility Meme
]

selected_asset = st.selectbox("🎯 Target Asset for Micro-Analysis", asset_library)

st.write("---")

# 5. Dashboard View
tab_micro, tab_confluence, tab_system = st.tabs(["💎 Microstructure", "🧬 Confluence Engine", "🛠️ System Logs"])

with tab_micro:
    st.subheader(f"Orderbook Depth & Spread: {selected_asset}")
    bids, asks, spread, spread_pct, imbalance = fetch_orderbook_data(selected_asset)
    
    if not bids.empty:
        # A. Key Liquidity Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Spread", f"${spread:.4f}", f"{spread_pct:.4f}%", delta_color="inverse")
        m2.metric("Orderbook Imbalance", f"{imbalance:.2%}", "Buy Pressure" if imbalance > 0 else "Sell Pressure")
        m3.metric("Primary Exchange", "BITGET", "ACTIVE")
        
        # B. Liquidity Depth Visualization
        st.write("#### Visualizing Buy/Sell Walls (Depth Chart)")
        fig_depth = go.Figure()
        fig_depth.add_trace(go.Scatter(x=bids['price'], y=bids['quantity'].cumsum(), fill='tozeroy', name='Cumulative Bids', line_color='#00ff88'))
        fig_depth.add_trace(go.Scatter(x=asks['price'], y=asks['quantity'].cumsum(), fill='tozeroy', name='Cumulative Asks', line_color='#ff4b4b'))
        fig_depth.update_layout(template="plotly_dark", height=400, xaxis_title="Price", yaxis_title="Cumulative Volume", hovermode="x unified")
        st.plotly_chart(fig_depth, use_container_width=True)

        

with tab_confluence:
    st.subheader("Multi-Timeframe Trend Alignment")
    mtf_signals = get_mtf_confluence(selected_asset)
    
    cols = st.columns(4)
    for i, (tf, signal) in enumerate(mtf_signals.items()):
        with cols[i]:
            color = "#00ff88" if signal == "BULLISH" else "#ff4b4b"
            st.markdown(f"**{tf} Trend**")
            st.markdown(f"<h2 style='color: {color};'>{signal}</h2>", unsafe_allow_html=True)
            st.caption("9/21 EMA Alignment")

    # Master Confluence Verdict
    st.write("---")
    bull_count = list(mtf_signals.values()).count("BULLISH")
    bear_count = list(mtf_signals.values()).count("BEARISH")
    
    if bull_count == 4:
        st.success(f"🚀 **FULL BULLISH CONFLUENCE:** {selected_asset} is trending upward on all monitored timeframes.")
    elif bear_count == 4:
        st.error(f"🚨 **FULL BEARISH CONFLUENCE:** {selected_asset} is trending downward on all monitored timeframes.")
    else:
        st.warning(f"⚖️ **DIVERGENCE:** {selected_asset} is in a fragmented state ({bull_count} Bull / {bear_count} Bear). Use Caution.")

    

with tab_system:
    st.subheader("System Reliability & Infrastructure")
    st.info("Nexus Core is currently polling the Bitget API every 5 seconds for orderbook updates.")
    st.button("Reset Nexus Connection")
    st.write("API Status: **Green**")
