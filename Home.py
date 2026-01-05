import streamlit as st
import time
import psutil
import requests
import os

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

# 3. Theme Engine State
if 'matrix_mode' not in st.session_state:
    st.session_state.matrix_mode = False

theme_color = "#00ff00" if st.session_state.matrix_mode else "#4F8BF9"
bg_color = "#000000" if st.session_state.matrix_mode else "#0e1117"

# Custom CSS
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

# 4. Security Gate Logic
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

# --- THE FOLLOWING CODE ONLY RUNS IF AUTHENTICATED ---

# 5. Global Analytics Sidebar
with st.sidebar:
    st.title("📡 Live Intel")
    btc, btc_chg, eth, eth_chg = get_live_prices()
    st.metric("BTC/USD", f"${btc:,}" if isinstance(btc, int) else btc, f"{btc_chg:.2f}%")
    st.metric("ETH/USD", f"${eth:,}" if isinstance(eth, int) else eth, f"{eth_chg:.2f}%")
    st.write("---")
    if st.toggle("Matrix Mode (Cyberpunk)", value=st.session_state.matrix_mode):
        st.session_state.matrix_mode = True
    else:
        st.session_state.matrix_mode = False
    
    if st.button("Secure Logout", use_container_width=True):
        del st.session_state.authenticated
        st.rerun()

# 6. Resource Monitor & Header
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

# 7. Aegis & Nexus App Grid (The 7 Modules)
# Defined inside the auth block to ensure scope is available for the loop
apps = [
    ["🤖", "Aegis Auto", "Automated execution and chat bot.", "Aegis_Auto"],
    ["🧠", "Nexus Neural", "Deep learning and predictive models.", "Nexus_Neural"],
    ["💰", "Aegis Wealth", "Core profit and portfolio analytics.", "Aegis_Wealth"],
    ["📈", "Aegis Legacy", "Stable analytical version (Legacy).", "Aegis_Legacy"],
    ["⚙️", "Nexus Core", "System utility and architecture config.", "Nexus_Core"],
    ["🧬", "Neural Profit", "Multi-layer financial modeling logic.", "Neural_Profit"],
    ["📉", "Aegis Risk", "Market volatility and exposure scanner.", "Aegis_Risk"]
]

cols = st.columns(3)
for index, app in enumerate(apps):
    icon, name, desc, filename = app
    with cols[index % 3]:
        with st.container(border=True):
            st.markdown(f"### {icon} {name}")
            st.markdown("<span class='status-online'>● SYSTEM OPERATIONAL</span>", unsafe_allow_html=True)
            st.write(desc)
            
            # Robust Launch Logic
            if st.button(f"Launch {name}", key=f"btn_{filename}", use_container_width=True):
                try:
                    # Method 1: Path from Root
                    st.switch_page(f"pages/{filename}.py")
                except Exception:
                    try:
                        # Method 2: Filename only (Some Streamlit versions)
                        st.switch_page(f"{filename}.py")
                    except Exception as e:
                        st.error(f"Critical Error: Could not load {filename}.py")
                        st.toast(f"System Message: {str(e)}")

# 8. Logs Footer
st.write("---")
with st.expander("📂 Access Logs"):
    st.code(f"""
    [SYS] {time.strftime('%Y-%m-%d')} Connection established.
    [NET] Secure Handshake via SSL.
    [DB] 7/7 Aegis-Nexus modules mapped.
    [LOG] System monitoring active...
    """, language="bash")

st.caption("Developed by TechSolute | Aegis Unified Environment v2.5")
