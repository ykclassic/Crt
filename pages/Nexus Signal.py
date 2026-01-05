import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Neural Profit | Multi-Timeframe", page_icon="🧬", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.warning("⚠️ Access Denied. Redirecting to Command Center...")
    st.switch_page("Home.py")
    st.stop()

# 3. Theme Sync
theme_color = "#00ff00" if st.session_state.get('matrix_mode', False) else "#4F8BF9"

# 4. Global Aegis Header
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title("🧬 Neural Profit Engine")
    st.write(f"Strategy: **Multi-Timeframe Trend Confirmation (1D -> 4H -> 1H)**")
with col_h2:
    if st.button("🏠 Back to Home", use_container_width=True):
        st.switch_page("Home.py")

st.write("---")

# --- CORE ANALYTICS ENGINE ---

def fetch_safe_data(exchange_id, symbol, timeframe, limit=100):
    try:
        ex = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except:
        return None

def get_trend(df):
    """Simple EMA trend detection"""
    ema_20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
    last_price = df['close'].iloc[-1]
    return "BULLISH" if last_price > ema_20 else "BEARISH"

def calculate_levels(df, direction):
    """Calculates Entry, SL, and TP using ATR-style volatility"""
    last_price = df['close'].iloc[-1]
    # Calculate simple volatility (High-Low average)
    volatility = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    
    if direction == "BULLISH":
        entry = last_price
        sl = entry - (volatility * 1.5)
        tp = entry + (volatility * 3.0) # 1:2 Risk-Reward
    else:
        entry = last_price
        sl = entry + (volatility * 1.5)
        tp = entry - (volatility * 3.0)
    
    return round(entry, 4), round(sl, 4), round(tp, 4)

# --- UI LAYOUT ---

col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("🛠️ Analysis Parameters")
    with st.container(border=True):
        ex_id = st.selectbox("Exchange", ["bitget", "gateio", "xt"])
        asset = st.text_input("Asset (e.g. BTC, ETH)", value="BTC").upper()
        pair = f"{asset}/USDT"
        
        if st.button("⚡ Generate Neural Signals", use_container_width=True):
            with st.spinner("Analyzing Timeframe Confluence..."):
                # Fetch 3 Timeframes
                df_1d = fetch_safe_data(ex_id, pair, '1d')
                df_4h = fetch_safe_data(ex_id, pair, '4h')
                df_1h = fetch_safe_data(ex_id, pair, '1h')
                
                if df_1d is not None and df_4h is not None and df_1h is not None:
                    trend_1d = get_trend(df_1d)
                    trend_4h = get_trend(df_4h)
                    
                    st.session_state.signal_data = {
                        "df": df_1h,
                        "trend_1d": trend_1d,
                        "trend_4h": trend_4h,
                        "pair": pair
                    }
                    
                    # Confirm Direction
                    if trend_1d == trend_4h:
                        entry, sl, tp = calculate_levels(df_1h, trend_1d)
                        st.session_state.signal_data.update({"entry": entry, "sl": sl, "tp": tp, "status": "CONFIRMED"})
                    else:
                        st.session_state.signal_data.update({"status": "CONFLICT"})

with col_output:
    st.subheader("📊 Neural Recommendations")
    if "signal_data" in st.session_state:
        sd = st.session_state.signal_data
        
        # Trend Badges
        c1, c2, c3 = st.columns(3)
        c1.write(f"**1D Trend:** {'🟢' if sd['trend_1d'] == 'BULLISH' else '🔴'} {sd['trend_1d']}")
        c2.write(f"**4H Trend:** {'🟢' if sd['trend_4h'] == 'BULLISH' else '🔴'} {sd['trend_4h']}")
        
        if sd['status'] == "CONFIRMED":
            st.success(f"✅ CONFLUENCE DETECTED: Market bias is {sd['trend_1d']}")
            
            # Recommendation Box
            with st.container(border=True):
                r1, r2, r3 = st.columns(3)
                r1.metric("ENTRY (1H)", sd['entry'])
                r2.metric("STOP LOSS", sd['sl'], delta_color="inverse")
                r3.metric("TAKE PROFIT", sd['tp'])
                
            # Chart with Levels
            fig = go.Figure(data=[go.Candlestick(x=sd['df']['ts'], open=sd['df']['open'], high=sd['df']['high'], low=sd['df']['low'], close=sd['df']['close'])])
            fig.add_hline(y=sd['entry'], line_dash="dash", line_color="white", annotation_text="ENTRY")
            fig.add_hline(y=sd['sl'], line_dash="dot", line_color="red", annotation_text="SL")
            fig.add_hline(y=sd['tp'], line_dash="dot", line_color="green", annotation_text="TP")
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ NO CONFLUENCE: 1D and 4H timeframes are in conflict. Avoid entry.")
    else:
        st.info("Run analysis to see neural recommendations based on timeframe confluence.")

# 9. Diagnostic Logs
st.write("---")
with st.expander("🔬 Multi-Timeframe Logic"):
    st.markdown("""
    - **1D Filter:** Ensures you aren't trading against the primary market direction.
    - **4H Confirmation:** Confirms intermediate momentum is aligned with the daily bias.
    - **1H Execution:** Uses volatility (ATR) to set Stop Loss and Take Profit levels that respect current market noise.
    """)
