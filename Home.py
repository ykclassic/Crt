import streamlit as st
import time

# MUST be the first streamlit command
st.set_page_config(
    page_title="Forge Command Center",
    page_icon="🚀",
    layout="wide"
)

# --- CSS FOR UI ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display:none;}
    
    .status-online {
        color: #00ff00;
        font-size: 0.8rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    /* Simple styling for the fake log */
    .log-text {
        font-family: 'Courier New', Courier, monospace;
        color: #00ff00;
        background-color: #000;
        padding: 5px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BOOT SEQUENCE ANIMATION ---
# This runs only when the app starts
with st.status("🚀 Initializing Forge Ecosystem...", expanded=True) as status:
    st.write("Checking connection to database...")
    time.sleep(0.4)
    st.write("Loading Machine Learning modules (ForgeML)...")
    time.sleep(0.4)
    st.write("Verifying ProfitForge analytics engine...")
    time.sleep(0.4)
    st.write("System Integrity Check: 100%")
    status.update(label="✅ System Ready. All Modules Online.", state="complete", expanded=False)

# --- MAIN DASHBOARD CONTENT ---
st.title("📟 Forge Command Center")
st.write(f"Last Boot: {time.strftime('%H:%M:%S')} | Network: **Stable**")
st.write("---")

# Data for your 7 apps
apps = [
    ["🤖", "ForgeBot", "Automated bot for task execution.", "ForgeBot", "Online"],
    ["🧠", "ForgeML", "Machine Learning models.", "ForgeML", "Online"],
    ["💰", "ProfitForge", "Core profit tracking tool.", "ProfitForge", "Online"],
    ["📈", "ProfitForgev1", "Legacy stable analytics.", "ProfitForgev1", "Stable"],
    ["⚙️", "forge5.2", "System utility and config.", "forge5.2", "Online"],
    ["🧬", "profitforge_mlp", "Neural network logic.", "profitforge_mlp", "Online"],
    ["📉", "volatility_scanner", "Market risk scanner.", "volatility_scanner", "Live"]
]

# --- GRID LAYOUT ---
cols = st.columns(3)

for index, app in enumerate(apps):
    icon, name, desc, filename, s_text = app
    with cols[index % 3]:
        with st.container(border=True):
            st.markdown(f"### {icon} {name}")
            st.markdown(f"<span class='status-online'>● {s_text}</span>", unsafe_allow_html=True)
            st.write(desc)
            
            if st.button(f"Launch {name}", key=f"btn_{name}", use_container_width=True):
                try:
                    st.switch_page(f"pages/{filename}.py")
                except:
                    st.error(f"Error: pages/{filename}.py not found.")

st.write("---")

# --- TERMINAL LOG FOOTER ---
with st.expander("📂 View System Logs"):
    st.code(f"""
    [SYS_INFO] User Connected: Remote Access
    [MODULES] 7/7 Apps loaded successfully
    [ENV] Secure Environment Active
    [LOG] {time.strftime('%Y-%m-%d %H:%M:%S')} - Waiting for command...
    """, language="bash")

st.caption("Developed by TechSolute | Forge Unified Environment v2.2")
