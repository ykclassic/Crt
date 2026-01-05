import streamlit as st
import time
import psutil
import requests
import sqlite3
import pandas as pd
from datetime import datetime
import os

# 1. Page Configuration
st.set_page_config(page_title="Aegis OS", page_icon="🛡️", layout="wide")

# 2. Database Integration
def init_db():
    conn = sqlite3.connect('aegis_system.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs 
                 (timestamp TEXT, user_level TEXT, event TEXT)''')
    conn.commit()
    conn.close()

def add_log(user_level, event):
    conn = sqlite3.connect('aegis_system.db', check_same_thread=False)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO logs VALUES (?, ?, ?)", (now, user_level, event))
    conn.commit()
    conn.close()

def get_logs():
    try:
        conn = sqlite3.connect('aegis_system.db', check_same_thread=False)
        df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame(columns=["timestamp", "user_level", "event"])

init_db()

# 3. Live Data
def get_live_prices():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=5).json()
        return response['bitcoin']['usd'], response['bitcoin']['usd_24h_change']
    except:
        return "N/A", 0

# 4. Auth Logic
PASSCODES = {"admin123": "Admin", "analyst456": "Analyst", "view789": "Observer"}

if "authenticated" not in st.session_state:
    st.title("🛡️ Aegis Secure Terminal")
    pwd = st.text_input("Enter Authorization Key:", type="password")
    if st.button("Access System"):
        if pwd in PASSCODES:
            st.session_state.authenticated = True
            st.session_state.user_level = PASSCODES[pwd]
            add_log(st.session_state.user_level, "System Login")
            st.rerun()
    st.stop()

# 5. Theme
st.markdown("<style>.stApp {background-color: #0e1117;}</style>", unsafe_allow_html=True)

# 6. Sidebar
with st.sidebar:
    st.title("📡 Live Intel")
    btc, btc_chg = get_live_prices()
    st.metric("BTC/USD", f"${btc:,}" if isinstance(btc, int) else btc, f"{btc_chg:.2f}%")
    st.write("---")
    st.write(f"Identity: **{st.session_state.user_level}**")
    if st.button("Logout", use_container_width=True):
        del st.session_state.authenticated
        st.rerun()

# 7. Header
st.title("🛡️ Aegis Command Center")
st.write(f"Environment: Production | {time.strftime('%H:%M:%S')}")
st.write("---")

# 8. App Grid
apps = [
    ["🤖", "Aegis Auto", "Aegis_Auto", ["Admin", "Analyst"]],
    ["🧠", "Nexus Neural", "Nexus_Neural", ["Admin", "Analyst"]],
    ["💰", "Aegis Wealth", "Aegis_Wealth", ["Admin", "Analyst", "Observer"]],
    ["📡", "Nexus Signal", "Nexus_Signal", ["Admin", "Analyst"]],
    ["⚙️", "Nexus Core", "Nexus_Core", ["Admin"]],
    ["🧬", "Neural Profit", "Neural_Profit", ["Admin", "Analyst"]],
    ["📉", "Aegis Risk", "Aegis_Risk", ["Admin", "Analyst", "Observer"]]
]

visible_apps = [a for a in apps if st.session_state.user_level in a[3]]
cols = st.columns(3)
for index, app in enumerate(visible_apps):
    icon, name, filename, roles = app
    with cols[index % 3]:
        with st.container(border=True):
            st.markdown(f"### {icon} {name}")
            if st.button(f"Launch {name}", key=f"btn_{filename}", use_container_width=True):
                add_log(st.session_state.user_level, f"Launched {name}")
                st.switch_page(f"pages/{filename}.py")

st.caption("Aegis Unified Environment v3.3")
