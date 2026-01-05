import streamlit as st
import time
import psutil
import requests

# 1. Page Configuration
st.set_page_config(page_title="Aegis OS", page_icon="🛡️", layout="wide")

# 2. Live Data Fetcher
def get_live_prices():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url).json()
        btc = response['bitcoin']['usd']
        btc_chg = response['bitcoin']['usd_24h_change']
        eth = response['ethereum']['usd']
        eth_chg = response['ethereum']['usd_24h_change']
        return btc, btc_chg, eth, eth_chg
    except:
        return "N/A", 0, "N/A", 0

# 3. Security Gate (Feature 1)
if "authenticated" not in st.session_state:
    st.title("🛡️ Aegis Secure Terminal")
    pwd = st.text_input("Enter Authorization Key:", type="password")
    if st.button("Access System"):
        if pwd == "forge2026":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Access Denied: Invalid Key")
    st.stop()

# 4. Theme Engine (Feature 5)
if 'matrix_mode' not in st.session_state:
    st.session_state.matrix_mode = False

theme_color = "#00ff00" if st.session_state.matrix_mode else "#4F8BF9"
bg_color = "#000000" if st.session_state.matrix_mode else "#0e1117"

st.markdown(f"""
    <style>
    footer {{visibility: hidden;}}
    .stApp {{background-color: {bg_color};}}
    .status-online {{ color: {theme_color}; font-weight: bold; font-size: 0.8rem; }}
    .stMetric label {{ color: {theme_color} !important; }}
    .stButton>button {{ border: 1px solid {theme_color}; color: {theme_color}; background-color: transparent; }}
    .stButton>button:hover {{ background-color: {theme_color}; color: black; }}
    </style>
    """, unsafe_allow_html=True)

# 5. Sidebar - Live Intel & Controls (Feature 3)
with st.sidebar:
    st.title("📡 Live Intel")
    btc, btc_chg, eth, eth_chg = get_live_prices()
    st.metric("BTC/USD", f"${btc:,}" if isinstance(btc, int) else btc, f"{btc_chg:.2f}%")
    st.metric("ETH/USD", f"${eth:,}" if isinstance(eth, int) else eth, f"{eth_chg:.2f}%")
    st.write("---")
    if st.toggle("Matrix Mode (Cyberpunk)"):
        st.session_state.matrix_mode = True
    else:
        st.session_state.matrix_mode = False
    if st.button("Secure Logout"):
        del st.session_state.authenticated
        st.rerun()

# 6. Resource Monitor & Header (Feature 6)
col_main, col_res = st.columns([4, 1])

with col_main:
    st.title("🛡️ Aegis Command Center")
    st.write(f"Session Active | Time: {time.strftime('%H:%M:%S')} | Env: **Production**")

with col_res:
    with st.container(border=True):
        cpu = psutil.cpu_percent()
        st.write(f"💻 CPU: {cpu}%")
        st.progress(cpu/100)
        mem = psutil.virtual_memory().percent
        st.write(f"🧠 RAM: {mem}%")
        st.progress(mem/100)

st.write("---")

import os

# --- TEMPORARY DEBUGGER ---
st.write("### 📂 Files Detected in /pages:")
try:
    files = os.listdir('pages')
    st.code(files)
except Exception as e:
    st.error(f"Cannot access pages folder: {e}")
# -------------------------

# 7. Aegis & Nexus App Grid (The 7 Modules)
# Format: [Icon, Display Name, Description, Filename]
# --- ROBUST GRID LAYOUT ---
# --- ROBUST GRID LAYOUT ---
cols = st.columns(3)

for index, app in enumerate(apps):
    icon, name, desc, filename = app
    with cols[index % 3]:
        with st.container(border=True):
            st.markdown(f"### {icon} {name}")
            st.markdown("<span class='status-online'>● SYSTEM OPERATIONAL</span>", unsafe_allow_html=True)
            st.write(desc)
            
            # THE ROBUST LAUNCHER
            if st.button(f"Launch {name}", key=f"btn_{filename}", use_container_width=True):
                # We try three different ways to hit the path to bypass Streamlit cache issues
                paths_to_try = [
                    f"pages/{filename}.py", 
                    f"{filename}.py", 
                    filename
                ]
                
                success = False
                for path in paths_to_try:
                    try:
                        st.switch_page(path)
                        success = True
                        break
                    except:
                        continue
                
                if not success:
                    st.error(f"Critical: {filename} not found in system map.")
                    st.toast("Try rebooting the app from the Streamlit Cloud dashboard.")

# 8. Logs
st.write("---")
with st.expander("📂 Access Logs"):
    st.code(f"""
    [SYS] {time.strftime('%Y-%m-%d')} Connection established.
    [NET] Secure Handshake via SSL.
    [DB] 7/7 Aegis-Nexus modules mapped.
    [LOG] System monitoring active...
    """, language="bash")
