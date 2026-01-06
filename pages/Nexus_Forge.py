import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import importlib.util
import os
import time

# 1. Page Configuration
st.set_page_config(page_title="Nexus Forge | Executive Node", page_icon="⚙️", layout="wide")

# 2. Security Check
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# --- THE AGGREGATION ENGINE ---
def poll_module(module_name, asset):
    """Dynamically imports and runs the core logic of a specific page."""
    path = f"pages/{module_name}.py"
    if not os.path.exists(path):
        return "OFFLINE", 0.0, "Path Error"
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        if hasattr(mod, "get_live_signal"):
            # Expecting return: (direction, confidence, timestamp)
            return mod.get_live_signal(asset)
    except Exception as e:
        return "ERROR", 0.0, str(e)
    return "UNKNOWN", 0.0, "No function"

def compute_executive_decision(data):
    """
    Safely calculates weighted consensus. 
    Handles unexpected verdicts without crashing.
    """
    weights = {"Nexus_Neural": 0.50, "Nexus_Signal": 0.30, "Neural_Profit": 0.20}
    total_weighted_conf = 0
    # Initialize votes with standard keys
    votes = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0, "OTHER": 0}
    
    for _, row in data.iterrows():
        # 1. Apply Weighting
        w = weights.get(row['Module'], 0.10)
        total_weighted_conf += row['Confidence'] * w
        
        # 2. SAFE VOTING: Check if verdict exists in dictionary, else use 'OTHER'
        v = str(row['Verdict']).upper()
        if v in votes:
            votes[v] += 1
        else:
            votes["OTHER"] += 1

    # Logic for Master Recommendation
    if total_weighted_conf > 85:
        decision = "🚀 STRONG EXECUTION"
        color = "#00FFCC" # Neon Green
    elif total_weighted_conf > 65:
        decision = "⚖️ ACCUMULATE / CAUTION"
        color = "#FFA500" # Orange
    else:
        decision = "📡 STANDBY / NEUTRAL"
        color = "#808080" # Grey
        
    return decision, round(total_weighted_conf, 2), color

# --- UI INTERFACE ---
st.title("⚙️ Nexus Forge: Executive Decision Node")
st.write("Aggregating multi-module intelligence into a single execution command.")

target_asset = st.selectbox("🎯 Target Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"])

if st.button("🛰️ Initialize Global Aggregation"):
    with st.status("Gathering Intelligence...", expanded=True) as status:
        # Modules to check in your /pages folder
        modules = ["Nexus_Neural", "Nexus_Signal", "Neural_Profit"]
        results = []
        
        for m in modules:
            st.write(f"Querying {m}...")
            signal = poll_module(m, target_asset)
            
            # Unpack results safely
            direction, conf, _ = signal
            results.append({"Module": m, "Verdict": direction, "Confidence": conf})
        
        df_results = pd.DataFrame(results)
        status.update(label="Aggregation Complete", state="complete", expanded=False)

    # MASTER VERDICT
    decision, final_conf, theme_color = compute_executive_decision(df_results)
    
    # Large Executive Banner
    st.markdown(f"""
        <div style="background-color:{theme_color}; padding:25px; border-radius:15px; text-align:center; border: 2px solid white;">
            <h1 style="color:black; margin:0; font-size:3rem;">{decision}</h1>
            <p style="color:black; font-weight:bold; font-size:1.5rem;">GLOBAL CONFIDENCE: {final_conf}%</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("---")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Module Confluence Map")
        # Radar/Bar Chart showing module strength
        fig = go.Figure(go.Bar(
            x=df_results['Module'], 
            y=df_results['Confidence'],
            marker_color=[theme_color] * len(df_results)
        ))
        fig.update_layout(template="plotly_dark", height=300, yaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Raw Intelligence Feed")
        st.dataframe(df_results, use_container_width=True)

st.caption(f"Nexus Forge v5.1 | Last Update: {time.strftime('%H:%M:%S')}")
