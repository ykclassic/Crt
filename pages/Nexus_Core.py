import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from cryptography.fernet import Fernet
import os
import sqlite3

def get_live_signal(asset="BTC/USDT"):
    # Insert your actual model logic here
    # Example mock return:
    return "LONG", 92.5, "12:00:00"

if __name__ == "__main__":
    # Your existing UI code goes inside this block
    # This prevents the UI from rendering when Nexus Forge imports the file
    pass 

# --- 1. SYSTEM SECURITY & DATABASE ---
def get_or_create_key():
    """Generates or retrieves the master encryption key."""
    if not os.path.exists(".vault_key"):
        key = Fernet.generate_key()
        with open(".vault_key", "wb") as f:
            f.write(key)
    return open(".vault_key", "rb").read()

def encrypt_val(data):
    f = Fernet(get_or_create_key())
    return f.encrypt(data.encode()).decode()

def decrypt_val(data):
    f = Fernet(get_or_create_key())
    return f.decrypt(data.encode()).decode()

def init_db():
    conn = sqlite3.connect("aegis_vault.db")
    conn.execute("CREATE TABLE IF NOT EXISTS vault (exchange TEXT PRIMARY KEY, key TEXT, secret TEXT)")
    conn.commit()
    conn.close()

init_db()

# --- 2. ENGINE LOGIC ---
@st.cache_data(ttl=5)
def get_arbitrage_node(symbol):
    """Calculates price spread between Bitget and XT."""
    try:
        p1 = ccxt.bitget().fetch_ticker(symbol)['last']
        p2 = ccxt.xt().fetch_ticker(symbol)['last']
        diff = abs(p1 - p2)
        pct = (diff / max(p1, p2)) * 100
        return p1, p2, diff, pct
    except: return 0, 0, 0, 0

def get_micro_depth(symbol):
    """Calculates orderbook walls and imbalance."""
    try:
        ex = ccxt.bitget()
        ob = ex.fetch_order_book(symbol, limit=25)
        bids = pd.DataFrame(ob['bids'], columns=['price', 'quantity'])
        asks = pd.DataFrame(ob['asks'], columns=['price', 'quantity'])
        imbalance = (bids['quantity'].sum() - asks['quantity'].sum()) / (bids['quantity'].sum() + asks['quantity'].sum())
        return bids, asks, imbalance
    except: return pd.DataFrame(), pd.DataFrame(), 0

# --- 3. UI INITIALIZATION ---
st.set_page_config(page_title="Nexus Core | Engine", layout="wide")

if "authenticated" not in st.session_state:
    st.switch_page("Home.py")

st.title("⚙️ Nexus Core: Integrated Infrastructure")

# Expanded Asset Library
assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT", "TRX/USDT", "SUI/USDT", "PEPE/USDT"]
selected = st.selectbox("🎯 Select Active Node", assets)

st.write("---")

tab_arb, tab_micro, tab_mtf, tab_vault = st.tabs(["⚖️ Arbitrage", "💎 Microstructure", "🧬 Confluence", "🔐 Secure Vault"])

# --- TAB: ARBITRAGE ---
with tab_arb:
    p_bg, p_xt, diff, pct = get_arbitrage_node(selected)
    c1, c2, c3 = st.columns(3)
    c1.metric("Bitget Price", f"${p_bg:,.4f}")
    c2.metric("XT.com Price", f"${p_xt:,.4f}")
    c3.metric("Spread (%)", f"{pct:.3f}%", delta=f"{diff:,.4f}")
    
    if pct > 0.15:
        st.success(f"🔥 ARBITRAGE ALERT: High spread detected on {selected}")

# --- TAB: MICROSTRUCTURE ---
with tab_micro:
    bids, asks, imb = get_micro_depth(selected)
    if not bids.empty:
        st.metric("Orderbook Imbalance", f"{imb:.2%}", "Buy Pressure" if imb > 0 else "Sell Pressure")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bids['price'], y=bids['quantity'].cumsum(), fill='tozeroy', name='Bids', line_color='#00ffcc'))
        fig.add_trace(go.Scatter(x=asks['price'], y=asks['quantity'].cumsum(), fill='tozeroy', name='Asks', line_color='#ff4b4b'))
        fig.update_layout(template="plotly_dark", height=400, title="Liquidity Depth Map")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB: CONFLUENCE ---
with tab_mtf:
    st.subheader("Multi-Timeframe Trend Alignment")
    # Simulation of MTF Engine logic
    tfs = ["15m", "1h", "4h", "1d"]
    cols = st.columns(4)
    for i, tf in enumerate(tfs):
        with cols[i]:
            st.write(f"### {tf}")
            st.info("EMA Alignment: STABLE")

# --- TAB: SECURE VAULT ---
with tab_vault:
    st.subheader("🔐 API Key Management (AES-256)")
    venue = st.radio("Exchange", ["Bitget", "XT.com"], horizontal=True)
    with st.form(f"vault_{venue}"):
        key_in = st.text_input("API Key", type="password")
        sec_in = st.text_input("API Secret", type="password")
        if st.form_submit_button("Encrypt & Store"):
            conn = sqlite3.connect("aegis_vault.db")
            conn.execute("INSERT OR REPLACE INTO vault VALUES (?, ?, ?)", (venue.lower(), encrypt_val(key_in), encrypt_val(sec_in)))
            conn.commit()
            conn.close()
            st.toast("Credentials Encrypted & Saved")

st.caption("Nexus Core v7.0 | Total System Integration")
