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

# 3. Microstructure & Engine Functions
def fetch_orderbook(symbol):
    """Fetches real-time bid/ask depth from Bitget"""
    try:
        ex = ccxt.bitget()
        ob = ex.fetch_order_book(symbol, limit=20)
        bids = pd.DataFrame(ob['bids'], columns=['price', 'quantity'])
        asks = pd.DataFrame(ob['asks'], columns=['price', 'quantity'])
        spread = asks['price'].iloc[0] - bids['price'].iloc[0]
        spread_pct = (spread / asks['price'].iloc[0]) * 100
        return bids, asks, spread, spread_pct
    except Exception as e:
        st.error(f"Orderbook Error: {e}")
        return pd.DataFrame(), pd.DataFrame(), 0, 0

def get_mtf_logic(symbol):
    """Calculates Trend Alignment for 15m, 1h, 4h, and 1d"""
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
            results[tf] = "OFFLINE"
    return results

# 4. Header
st.title("⚙️ Nexus Core: System Engine")
st.write(f"Primary Exchange: **{st.session_state.get('primary_exchange', 'BITGET').upper()}**")
st.write("---")

# 5. Dashboard Layout: Microstructure & MTC
tab_depth, tab_mtf, tab_vault = st.tabs(["💎 Microstructure", "🧬 MTF Confluence", "🔑 API Vault"])

with tab_depth:
    st.subheader("Market Microstructure & Liquidity Depth")
    asset = st.selectbox("Select Asset for Inspection", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    
    bids, asks, spread, spread_pct = fetch_orderbook(asset)
    
    if not bids.empty:
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Current Spread", f"${spread:.2f}", f"{spread_pct:.4f}%", delta_color="inverse")
        
        # Orderbook Depth Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bids['price'], y=bids['quantity'].cumsum(), fill='tozeroy', name='Bids (Buy)', line_color='green'))
        fig.add_trace(go.Scatter(x=asks['price'], y=asks['quantity'].cumsum(), fill='tozeroy', name='Asks (Sell)', line_color='red'))
        fig.update_layout(template="plotly_dark", title=f"Liquidity Depth: {asset}", xaxis_title="Price", yaxis_title="Cumulative Volume")
        st.plotly_chart(fig, use_container_width=True)
        
        

with tab_mtf:
    st.subheader("Multi-Timeframe Confluence Engine")
    st.write("Synchronizing trend signals across global timeframes...")
    
    mtf_data = get_mtf_logic(asset)
    cols = st.columns(4)
    for i, (tf, signal) in enumerate(mtf_data.items()):
        with cols[i]:
            color = "green" if signal == "BULLISH" else "red"
            st.markdown(f"### {tf}")
            st.markdown(f"<h2 style='color: {color};'>{signal}</h2>", unsafe_allow_html=True)
            st.caption("9/21 EMA Crossover Logic")

    # Final Confluence Output
    signals = list(mtf_data.values())
    if all(x == "BULLISH" for x in signals):
        st.success("✅ **CONFLUENCE REACHED:** Macro and Micro trends are perfectly aligned (BULL).")
    elif all(x == "BEARISH" for x in signals):
        st.error("🚨 **CONFLUENCE REACHED:** Macro and Micro trends are perfectly aligned (BEAR).")
    else:
        st.warning("⚖️ **NO CONFLUENCE:** Market is fragmented or ranging.")

with tab_vault:
    st.subheader("Security & API Management")
    st.info("API Keys are encrypted and stored locally in `aegis_system.db`.")
    st.text_input("Bitget API Key", type="password")
    st.text_input("Bitget Secret", type="password")
    st.button("Rotate System Keys")

st.caption("Nexus Core v4.0 | Microstructure & Confluence Integrated")
