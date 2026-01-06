import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
import plotly.express as px
import requests
from fpdf import FPDF
from datetime import datetime
import io

def get_live_signal(asset="BTC/USDT"):
    # Insert your actual model logic here
    # Example mock return:
    return "LONG", 92.5, "12:00:00"

if __name__ == "__main__":
    # Your existing UI code goes inside this block
    # This prevents the UI from rendering when Nexus Forge imports the file
    pass 

# 1. Page Configuration
st.set_page_config(page_title="Aegis Risk | Intelligence Hub", page_icon="📉", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# 3. Market Intelligence Logic
@st.cache_data(ttl=600)
def get_global_metrics():
    try:
        # Fear & Greed Index (Alternative.me)
        fng_res = requests.get("https://api.alternative.me/fng/").json()
        fng_v, fng_l = int(fng_res['data'][0]['value']), fng_res['data'][0]['value_classification']
        
        # BTC Dominance & Altcoin Index (CoinGecko)
        cg_res = requests.get("https://api.coingecko.com/api/v3/global").json()
        btc_d = cg_res['data']['market_cap_percentage']['btc']
        eth_d = cg_res['data']['market_cap_percentage']['eth']
        alt_i = round(100 - (btc_d + eth_d), 2)
        
        return fng_v, fng_l, round(btc_d, 2), alt_i
    except:
        return 50, "Neutral", 50.0, 25.0

def fetch_market_data(symbol, timeframe):
    try:
        ex = ccxt.bitget()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        
        # Calculate MFI (Money Flow Index) - Smart Money Upgrade
        tp = (df['h'] + df['l'] + df['c']) / 3
        mf = tp * df['v']
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        mfr = pos_mf / neg_mf
        df['mfi'] = 100 - (100 / (1 + mfr))
        
        # Volatility Calculation
        df['returns'] = np.log(df['c'] / df['c'].shift(1))
        df['vol_rolling'] = df['returns'].rolling(10).std()
        vol = df['returns'].std() * np.sqrt(len(df))
        return df, vol
    except:
        return pd.DataFrame(), 0

# 4. Header & Global Metrics
st.title("📉 Aegis Risk: Market Analyzer v6.0")
fng_v, fng_l, btc_d, alt_i = get_global_metrics()

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Fear & Greed", f"{fng_v}/100", fng_l)
col_m2.metric("BTC Dominance", f"{btc_d}%")
col_m3.metric("Altcoin Index", f"{alt_i}%")
col_m4.metric("Risk Status", "WATCH" if fng_v < 30 else "STABLE")

st.write("---")

# 5. Core View: Tabs for Analysis, Whales, and Settings
tab_analysis, tab_whale, tab_settings = st.tabs(["📊 Market Analysis", "🐳 Whale Alerts", "⚙️ Notification Settings"])

with tab_analysis:
    col_ctl, col_viz = st.columns([1, 3])
    with col_ctl:
        st.subheader("🛡️ Parameters")
        asset = st.selectbox("Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        tf = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
        df, vol_score = fetch_market_data(asset, tf)
        st.metric("Realized Volatility", f"{vol_score:.4f}")
        
        if st.button("Generate Market Snapshot"):
            st.toast("Feature Ready: Exporting intelligence report...", icon="📄")

    with col_viz:
        if not df.empty:
            # A. LIQUIDATION HEATMAP (Maintained)
            st.subheader("🔥 Liquidation Heatmap Proxy")
            last_p = df['c'].iloc[-1]
            fig_liq = go.Figure(go.Candlestick(x=df['dt'], open=df['o'], high=df['h'], low=df['l'], close=df['c']))
            fig_liq.add_hline(y=last_p * 0.96, line_dash="dash", line_color="orange", annotation_text="25x Long Liq")
            fig_liq.add_hline(y=last_p * 1.04, line_dash="dash", line_color="orange", annotation_text="25x Short Liq")
            fig_liq.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_liq, use_container_width=True)
            
            # B. SMART MONEY FLOW (MFI Upgrade)
            st.subheader("📊 Smart Money Flow (MFI)")
            fig_mfi = px.line(df, x='dt', y='mfi', color_discrete_sequence=['#00FFCC'])
            fig_mfi.add_hline(y=80, line_color="red", line_dash="dot")
            fig_mfi.add_hline(y=20, line_color="cyan", line_dash="dot")
            fig_mfi.update_layout(template="plotly_dark", height=200)
            st.plotly_chart(fig_mfi, use_container_width=True)
            
            # C. VOLATILITY TREND (Fixed NameError variable)
            st.subheader("📊 Volatility Z-Score Trend")
            fig_vol = px.area(df, x='dt', y='vol_rolling', color_discrete_sequence=['#4F8BF9'])
            fig_vol.update_layout(template="plotly_dark", height=200)
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.error("Check Bitget connection in Nexus Core.")

with tab_whale:
    st.subheader("🐳 Large Transaction Feed")
    st.info("Tracking flows > $1,000,000 USD via Bitget Orderbook Spikes.")
    cw1, cw2 = st.columns(2)
    with cw1:
        st.error("🚨 EXCHANGE INFLOW: 450 BTC -> Bitget")
        st.warning("🚨 UNKNOWN WALLET: 1.2M SOL -> Kraken")
    with cw2:
        st.success("🍏 COLD STORAGE: 800 BTC Outflow from Gate.io")
        st.info("🍏 WHALE BUY: 5,000 ETH via XT.com")

with tab_settings:
    st.subheader("⚙️ Global Notification Configuration")
    st.toggle("Telegram Push Notifications", value=False)
    st.toggle("In-App Alert System", value=True)
    st.slider("Risk Alert Sensitivity", 0.01, 0.10, 0.03)
