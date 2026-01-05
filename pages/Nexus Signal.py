import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(page_title="Nexus Signal | Aegis OS", page_icon="📡", layout="wide")

# 2. Security
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# 3. Header
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title("📡 Nexus Signal")
    st.write("Targeting System: **Multi-Timeframe Confluence (1D/4H/1H)**")
with col_h2:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

st.write("---")

# 4. Logic Functions
def fetch_safe_data(exchange_id, symbol, timeframe):
    try:
        ex = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching {timeframe} data: {e}")
        return None

def get_trend(df):
    ema_20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
    return "BULLISH" if df['close'].iloc[-1] > ema_20 else "BEARISH"

def calculate_levels(df, direction):
    last_price = df['close'].iloc[-1]
    volatility = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    if direction == "BULLISH":
        entry, sl, tp = last_price, last_price - (volatility * 1.5), last_price + (volatility * 3)
    else:
        entry, sl, tp = last_price, last_price + (volatility * 1.5), last_price - (volatility * 3)
    return round(entry, 4), round(sl, 4), round(tp, 4)

# 5. UI Layout
col_in, col_out = st.columns([1, 2])

with col_in:
    st.subheader("🛠️ Signal Filters")
    with st.container(border=True):
        ex_id = st.selectbox("Exchange", ["bitget", "gateio", "xt"])
        asset = st.text_input("Asset (e.g., BTC)", "BTC").upper() + "/USDT"
        
        if st.button("⚡ Generate Signal", use_container_width=True):
            with st.spinner("Analyzing Market Structure..."):
                d1 = fetch_safe_data(ex_id, asset, '1d')
                d4 = fetch_safe_data(ex_id, asset, '4h')
                d1h = fetch_safe_data(ex_id, asset, '1h')
                
                if d1 is not None and d4 is not None and d1h is not None:
                    t1, t4 = get_trend(d1), get_trend(d4)
                    st.session_state.nexus_sig = {"df": d1h, "t1": t1, "t4": t4, "asset": asset}
                    if t1 == t4:
                        e, s, t = calculate_levels(d1h, t1)
                        st.session_state.nexus_sig.update({"entry": e, "sl": s, "tp": t, "match": True})
                    else:
                        st.session_state.nexus_sig.update({"match": False})

with col_out:
    st.subheader("🎯 Recommendation")
    if "nexus_sig" in st.session_state:
        ns = st.session_state.nexus_sig
        
        c_t1, c_t4 = st.columns(2)
        c_t1.metric("1D Trend", ns['t1'])
        c_t4.metric("4H Trend", ns['t4'])
        
        if ns['match']:
            st.success(f"High-Probability {ns['t1']} Entry Found")
            r1, r2, r3 = st.columns(3)
            r1.metric("ENTRY", ns['entry'])
            r2.metric("STOP LOSS", ns['sl'], delta_color="inverse")
            r3.metric("TAKE PROFIT", ns['tp'])
            
            # Chart
            fig = go.Figure(data=[go.Candlestick(x=ns['df']['ts'], open=ns['df']['open'], high=ns['df']['high'], low=ns['df']['low'], close=ns['df']['close'])])
            fig.add_hline(y=ns['entry'], line_dash="dash", line_color="white", annotation_text="ENTRY")
            fig.add_hline(y=ns['sl'], line_dash="dot", line_color="red", annotation_text="SL")
            fig.add_hline(y=ns['tp'], line_dash="dot", line_color="green", annotation_text="TP")
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ NO CONFLUENCE: 1D and 4H trends are disconnected.")
    else:
        st.info("Awaiting market data request...")
