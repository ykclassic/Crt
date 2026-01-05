import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go

st.set_page_config(page_title="Nexus Signal", page_icon="📡", layout="wide")

if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# Header
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title("📡 Nexus Signal")
    st.write("Confluence: **1D Trend + 4H Confirmation**")
with col_h2:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("Home.py")

st.write("---")

def fetch_data(ex_id, asset, tf):
    try:
        ex = getattr(ccxt, ex_id)()
        data = ex.fetch_ohlcv(asset, timeframe=tf, limit=100)
        df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except: return None

def get_trend(df):
    ema = df['close'].ewm(span=20).mean().iloc[-1]
    return "BULLISH" if df['close'].iloc[-1] > ema else "BEARISH"

# Dashboard
col_in, col_out = st.columns([1, 2])
with col_in:
    with st.container(border=True):
        ex_id = st.selectbox("Exchange", ["bitget", "gateio", "xt"])
        sym = st.text_input("Asset", "BTC") + "/USDT"
        if st.button("Generate Strategy"):
            d1, d4, d1h = fetch_data(ex_id, sym, '1d'), fetch_data(ex_id, sym, '4h'), fetch_data(ex_id, sym, '1h')
            if d1 is not None and d4 is not None:
                t1, t4 = get_trend(d1), get_trend(d4)
                st.session_state.nexus_sig = {"df": d1h, "t1": t1, "t4": t4, "sym": sym}
                if t1 == t4:
                    last = d1h['close'].iloc[-1]
                    vol = (d1h['high'] - d1h['low']).iloc[-1]
                    st.session_state.nexus_sig.update({
                        "entry": last, 
                        "sl": last - (vol * 1.5) if t1 == "BULLISH" else last + (vol * 1.5),
                        "tp": last + (vol * 3) if t1 == "BULLISH" else last - (vol * 3),
                        "match": True
                    })
                else: st.session_state.nexus_sig.update({"match": False})

with col_out:
    if "nexus_sig" in st.session_state:
        ns = st.session_state.nexus_sig
        if ns['match']:
            st.success(f"Confluence Found: {ns['t1']}")
            st.metric("ENTRY", ns['entry'])
            st.metric("STOP LOSS", round(ns['sl'], 2))
            st.metric("TAKE PROFIT", round(ns['tp'], 2))
        else: st.error("No Confluence Found.")
    else: st.info("Run analysis to see target levels.")
