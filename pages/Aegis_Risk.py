import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import requests
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Aegis Risk | Market Intelligence", page_icon="📉", layout="wide")

# 2. Security
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# 3. Data Engine Functions
@st.cache_data(ttl=300)
def get_global_indices():
    """Fetches BTC Dominance and Altcoin Season Index via CoinGecko & Public API"""
    try:
        # BTC Dominance from CoinGecko
        cg_data = requests.get("https://api.coingecko.com/api/v3/global").json()
        btc_dom = cg_data['data']['market_cap_percentage']['btc']
        
        # Fear & Greed from Alternative.me
        fg_data = requests.get("https://api.alternative.me/fng/").json()
        fng_val = fg_data['data'][0]['value']
        fng_class = fg_data['data'][0]['value_classification']
        
        return round(btc_dom, 2), fng_val, fng_class
    except:
        return 0, 50, "Neutral"

def fetch_pair_volatility(symbol, tf):
    """Calculates Realized Volatility using Bitget Data"""
    try:
        ex = ccxt.bitget()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['returns'] = np.log(df['c'] / df['c'].shift(1))
        vol = df['returns'].std() * np.sqrt(len(df)) # Annualized Proxy
        return df, vol
    except:
        return pd.DataFrame(), 0

# 4. Header & Top Metrics (Dominance, Fear/Greed)
dom, fng_val, fng_text = get_global_indices()

st.title("📉 Aegis Risk: Market Intelligence")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("BTC Dominance", f"{dom}%")
col_m2.metric("Fear & Greed", f"{fng_val}/100", fng_text)
col_m3.metric("Altcoin Index", "42", "Neutral") # Hardcoded placeholder for Altcoin Season
col_m4.metric("Risk Status", "WATCH" if int(fng_val) > 70 else "SAFE")

st.write("---")

# 5. Volatility & Timeframe Analysis
col_side, col_main = st.columns([1, 3])

with col_side:
    st.subheader("⚙️ Analysis Params")
    asset = st.selectbox("Trading Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"])
    tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d", "1w", "1M"])
    
    df, vol_score = fetch_pair_volatility(asset, tf)
    st.metric(f"{tf} Realized Volatility", f"{vol_score:.2f}")
    
    if vol_score > 0.5:
        st.warning("⚠️ High Volatility Detected: Reduce Leverage")

with col_main:
    # 6. Liquidation Heatmap Proxy
    st.subheader(f"🔥 Liquidation Heatmap Proxy: {asset}")
    if not df.empty:
        last_price = df['c'].iloc[-1]
        # Simulate Liquidation Clusters at 10x, 25x, 50x leverage
        lev_levels = [0.90, 0.94, 0.96, 1.04, 1.06, 1.10] 
        heatmap_data = []
        
        fig = go.Figure()
        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df['ts'], open=df['o'], high=df['h'], low=df['l'], close=df['c'], name="Price"))
        
        # Add Heatmap "Danger Zones"
        colors = ['rgba(255, 0, 0, 0.2)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 255, 0, 0.2)']
        for i, mult in enumerate(lev_levels):
            level = last_price * mult
            fig.add_hline(y=level, line_dash="dot", line_color="orange", annotation_text=f"Liq Zone")

        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Connection to Bitget failed. Check Nexus Core API logs.")

# 7. Altcoin Index View
st.write("---")
st.subheader("🧬 Altcoin Rotation Index")
st.info("When BTC Dominance falls and Altcoin Index rises, capital is rotating to high-beta assets.")
# (Placeholder for more complex logic)
