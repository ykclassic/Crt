import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go

def get_live_signal(asset="BTC/USDT"):
    # Insert your actual model logic here
    # Example mock return:
    return "LONG", 92.5, "12:00:00"

# --- CORRECTION BELOW ---
if __name__ == "__main__":
    # Your existing UI code goes inside this block
    # This prevents the UI from rendering when Nexus Forge imports the file
    pass
# ------------------------
    
# 1. Page Config
st.set_page_config(page_title="Nexus Signal | Aegis OS", page_icon="📡", layout="wide")

# 2. Security
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# 3. Branding Sync
theme_color = "#00ff00" if st.session_state.get('matrix_mode', False) else "#4F8BF9"

# 4. Header
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title("📡 Nexus Signal")
    st.write("Targeting System: **Multi-Timeframe Confluence (1D/4H/1H)**")
with col_h2:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

st.write("---")

# 5. Logic Functions
def fetch_safe_data(exchange_id, symbol, timeframe):
    try:
        ex = getattr(ccxt, exchange_id)({'enableRateLimit': True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        # Add EMA for visual trend confirmation
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        return df
    except Exception as e:
        st.error(f"Error fetching {timeframe} data: {e}")
        return None

def get_trend(df):
    last_ema = df['ema_20'].iloc[-1]
    last_price = df['close'].iloc[-1]
    return "BULLISH" if last_price > last_ema else "BEARISH"

def calculate_levels(df, direction):
    last_price = df['close'].iloc[-1]
    # Volatility check using High-Low range of the last 14 candles
    volatility = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
    
    if direction == "BULLISH":
        entry = last_price
        sl = entry - (volatility * 1.5)
        tp = entry + (volatility * 3.0) # 1:2 Risk/Reward
    else:
        entry = last_price
        sl = entry + (volatility * 1.5)
        tp = entry - (volatility * 3.0)
    
    return round(entry, 4), round(sl, 4), round(tp, 4)

# 6. UI Layout
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
                    st.session_state.nexus_sig = {
                        "df": d1h, 
                        "t1": t1, 
                        "t4": t4, 
                        "asset": asset
                    }
                    if t1 == t4:
                        e, s, t = calculate_levels(d1h, t1)
                        st.session_state.nexus_sig.update({"entry": e, "sl": s, "tp": t, "match": True})
                    else:
                        st.session_state.nexus_sig.update({"match": False})

with col_out:
    st.subheader("🎯 Recommendation & Chart")
    if "nexus_sig" in st.session_state:
        ns = st.session_state.nexus_sig
        
        # Trend Status Badges
        c_t1, c_t4 = st.columns(2)
        c_t1.write(f"**1D Trend:** {'🟢' if ns['t1'] == 'BULLISH' else '🔴'} {ns['t1']}")
        c_t4.write(f"**4H Trend:** {'🟢' if ns['t4'] == 'BULLISH' else '🔴'} {ns['t4']}")
        
        if ns['match']:
            st.success(f"High-Probability {ns['t1']} Entry Found")
            
            # Key Metrics
            r1, r2, r3 = st.columns(3)
            r1.metric("ENTRY", ns['entry'])
            r2.metric("STOP LOSS", ns['sl'], delta_color="inverse")
            r3.metric("TAKE PROFIT", ns['tp'])
            
            # Interactive Chart
            df = ns['df']
            fig = go.Figure()
            
            # Candlesticks
            fig.add_trace(go.Candlestick(
                x=df['ts'], open=df['open'], high=df['high'], 
                low=df['low'], close=df['close'], name="Market Data"
            ))
            
            # EMA 20 Line
            fig.add_trace(go.Scatter(x=df['ts'], y=df['ema_20'], line=dict(color='gray', width=1), name="EMA 20"))
            
            # Signal Levels
            fig.add_hline(y=ns['entry'], line_dash="dash", line_color="white", annotation_text="ENTRY")
            fig.add_hline(y=ns['sl'], line_dash="dot", line_color="red", annotation_text="SL")
            fig.add_hline(y=ns['tp'], line_dash="dot", line_color="green", annotation_text="TP")
            
            fig.update_layout(
                template="plotly_dark", 
                height=450, 
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("⚠️ NO CONFLUENCE: 1D and 4H trends are disconnected. No signal generated.")
    else:
        st.info("Select parameters and click Generate to analyze the 1H execution chart.")

st.write("---")
st.caption("Nexus Signal v1.1 | Volatility-Adjusted Target Calculation")
