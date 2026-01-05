import streamlit as st
import time
import psutil
import requests
import sqlite3
import pandas as pd
from datetime import datetime

# 1. Page Configuration
st.set_page_config(page_title="Aegis OS", page_icon="🛡️", layout="wide")

# 2. Database Integration (Feature 4)
def init_db():
    conn = sqlite3.connect('aegis_system.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (timestamp TEXT, user_level TEXT, event TEXT)''')
    conn.commit()
    conn.close()

def add_log(user_level, event):
    conn = sqlite3.connect('aegis_system.db')
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO logs VALUES (?, ?, ?)", (now, user_level, event))
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect('aegis_system.db')
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def clear_logs():
    conn = sqlite3.connect('aegis_system.db')
    c = conn.cursor()
    c.execute("DELETE FROM logs")
    conn.commit()
    conn.close()

init_db()

# 3. Live Data Fetcher
def get_live_prices():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url).json()
        btc = response['bitcoin']['usd']
        btc_chg = response['bitcoin']['usd_24h_change']
        return btc, btc_chg
    except:
        return "N/A", 0

# 4. Auth Logic & User Levels
PASSCODES = {
    "admin123": "Admin",
    "analyst456": "Analyst",
    "view789": "Observer"
}

if "authenticated" not in st.session_state:
    st.title("🛡️ Aegis Secure Terminal")
    pwd = st.text_input("Enter Authorization Key:", type="password")
    if st.button("Access System"):
        if pwd in PASSCODES:
            st.session_state.authenticated = True
            st.session_state.user_level = PASSCODES[pwd]
            add_log(st.session_state.user_level, "System Login")
            st.toast(f"Access Granted: {st.session_state.user_level} Level", icon="🔐")
            st.rerun()
        else:
            st.error("Access Denied: Invalid Key")
    st.stop()

# 5. Theme Engine
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

# 6. Global Sidebar
with st.sidebar:
    st.title("📡 Live Intel")
    btc, btc_chg = get_live_prices()
    st.metric("BTC/USD", f"${btc:,}" if isinstance(btc, int) else btc, f"{btc_chg:.2f}%")
    
    if abs(btc_chg) > 2:
        st.toast(f"Volatility Alert: BTC moved {btc_chg:.2f}%", icon="⚠️")

    st.write("---")
    st.write(f"Identity: **{st.session_state.user_level}**")
    
    if st.toggle("Matrix Mode", value=st.session_state.matrix_mode):
        st.session_state.matrix_mode = True
    else:
        st.session_state.matrix_mode = False
    
    if st.button("Secure Logout", use_container_width=True):
        add_log(st.session_state.user_level, "System Logout")
        del st.session_state.authenticated
        st.rerun()

# 7. Main Dashboard Header
col_main, col_res = st.columns([4, 1])
with col_main:
    st.title("🛡️ Aegis Command Center")
    st.write(f"Environment: Production | {time.strftime('%H:%M:%S')}")

with col_res:
    with st.container(border=True):
        cpu = psutil.cpu_percent()
        st.write(f"💻 CPU: {cpu}%")
        st.progress(cpu/100)

# 8. Tabs for Navigation (Admin gets the Control Panel)
if st.session_state.user_level == "Admin":
    tab_apps, tab_admin = st.tabs(["🚀 Modules", "🛡️ Admin Control"])
else:
    tab_apps = st.container() # Non-admins don't see tabs

# MODULES TAB
with tab_apps:
    st.write("---")
    apps = [
        ["🤖", "Aegis Auto", "Automated execution bot.", "Aegis_Auto", ["Admin", "Analyst"]],
        ["🧠", "Nexus Neural", "Deep learning models.", "Nexus_Neural", ["Admin", "Analyst"]],
        ["💰", "Aegis Wealth", "Wealth analytics.", "Aegis_Wealth", ["Admin", "Analyst", "Observer"]],
        ["📈", "Aegis Legacy", "Stable version.", "Aegis_Legacy", ["Admin"]],
        ["⚙️", "Nexus Core", "System config.", "Nexus_Core", ["Admin"]],
        ["🧬", "Neural Profit", "Financial logic.", "Neural_Profit", ["Admin", "Analyst"]],
        ["📉", "Aegis Risk", "Exposure scanner.", "Aegis_Risk", ["Admin", "Analyst", "Observer"]]
    ]

    visible_apps = [a for a in apps if st.session_state.user_level in a[4]]
    cols = st.columns(3)
    
    for index, app in enumerate(visible_apps):
        icon, name, desc, filename, roles = app
        with cols[index % 3]:
            with st.container(border=True):
                st.markdown(f"### {icon} {name}")
                st.markdown("<span class='status-online'>● ACTIVE</span>", unsafe_allow_html=True)
                st.write(desc)
                if st.button(f"Launch {name}", key=f"btn_{filename}", use_container_width=True):
                    add_log(st.session_state.user_level, f"Launched {name}")
                    st.switch_page(f"pages/{filename}.py")

# ADMIN CONTROL TAB (Hidden from others)
if st.session_state.user_level == "Admin":
    with tab_admin:
        st.subheader("📊 System-Wide Usage Logs")
        log_df = get_logs()
        st.dataframe(log_df, use_container_width=True)
        
        col_c1, col_c2 = st.columns([1, 4])
        with col_c1:
            if st.button("🗑️ Clear All Logs", type="secondary"):
                clear_logs()
                st.success("Log database wiped.")
                st.rerun()
        st.info("The database records all logins, logouts, and module launches.")

st.write("---")
st.caption("Aegis Unified Environment v3.1 | Secure Session Active")
