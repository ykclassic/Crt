import streamlit as st
import time
import psutil
import requests  # To fetch live data

# 1. Page Configuration
st.set_page_config(page_title="Forge OS", page_icon="📟", layout="wide")

# 2. Function to fetch real-time Crypto Prices (Feature 3 Expanded)
def get_live_prices():
    try:
        # Fetching BTC and ETH from a public API (CoinGecko)
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url).json()
        btc_price = response['bitcoin']['usd']
        btc_change = response['bitcoin']['usd_24h_change']
        eth_price = response['ethereum']['usd']
        eth_change = response['ethereum']['usd_24h_change']
        return btc_price, btc_change, eth_price, eth_change
    except:
        return "N/A", 0, "N/A", 0

# 3. Theme & UI Styling
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
    </style>
    """, unsafe_allow_html=True)

# 4. Authentication Logic
if "authenticated" not in st.session_state:
    st.title("🔒 Forge Secure Access")
    pwd = st.text_input("Enter Command Center Passcode:", type="password")
    if st.button("Unlock Terminal"):
        if pwd == "forge2026":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Credentials")
    st.stop()

# --- CONTENT BELOW ONLY LOADS AFTER AUTHENTICATION ---

# 5. Global Analytics Sidebar (Live Data)
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
    
    if st.button("Logout"):
        del st.session_state.authenticated
        st.rerun()

# 6. Resource Monitor & Header
col_main, col_res = st.columns([4, 1])

with col_main:
    st.title("📟 Forge Command Center")
    st.write(f"System Time: {time.strftime('%H:%M:%S')} | Network: **Encrypted**")

with col_res:
    with st.container(border=True):
        cpu = psutil.cpu_percent()
        st.write(f"💻 CPU: {cpu}%")
        st.progress(cpu/100)
        mem = psutil.virtual_memory().percent
        st.write(f"🧠 RAM: {mem}%")
        st.progress(mem/100)

st.write("---")

# 7. Application Grid (7 Apps)
apps = [
    ["🤖", "ForgeBot", "Automated execution.", "ForgeBot"],
    ["🧠", "ForgeML", "Predictive models.", "ForgeML"],
    ["💰", "ProfitForge", "Profit analytics.", "ProfitForge"],
    ["📈", "ProfitForgev1", "Legacy stable version.", "ProfitForgev1"],
    ["⚙️", "forge5.2", "System config.", "forge5.2"],
    ["🧬", "profitforge_mlp", "Neural logic.", "profitforge_mlp"],
    ["📉", "volatility_scanner", "Market risk.", "volatility_scanner"]
]

cols = st.columns(3)
for index, app in enumerate(apps):
    icon, name, desc, filename = app
    with cols[index % 3]:
        with st.container(border=True):
            st.markdown(f"### {icon} {name}")
            st.markdown("<span class='status-online'>● OPERATIONAL</span>", unsafe_allow_html=True)
            st.write(desc)
            if st.button(f"Launch Module", key=f"btn_{name}", use_container_width=True):
                st.switch_page(f"pages/{filename}.py")

# 8. Footer Logs
st.write("---")
with st.expander("📂 Security Logs"):
    st.code(f"""
    [API] Connected to CoinGecko Feed... Success
    [AUTH] User session validated
    [RES] Monitoring CPU/RAM usage
    [LOG] {time.strftime('%Y-%m-%d %H:%M:%S')} - System Idle
    """, language="bash")
