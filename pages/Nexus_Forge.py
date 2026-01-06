import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import importlib.util
import os
import time

# 1. Page Config
st.set_page_config(page_title="Nexus Forge | Master Node", page_icon="⚙️", layout="wide")

if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# --- ROBUST MODULE PULLER ---
def get_module_recommendation(page_name, asset):
    """Safely pulls the get_live_signal function from sub-modules."""
    path = f"pages/{page_name}.py"
    if not os.path.exists(path):
        return "OFFLINE", 0.0, "Path Error"
    
    try:
        spec = importlib.util.spec_from_file_location(page_name, path)
        mod = importlib.util.module_from_spec(spec)
        # We wrap the execution to prevent sub-module UI from breaking the import
        spec.loader.exec_module(mod)
        
        if hasattr(mod, "get_live_signal"):
            direction, conf, _ = mod.get_live_signal(asset)
            return direction, conf, "Success"
        else:
            return "MISSING_FUNC", 0.0, "Function not found"
    except Exception as e:
        return "ERROR", 0.0, str(e)

# --- MASTER DECISION ENGINE ---
def generate_recommendation(df):
    """Analyzes aggregated data to give an informed final decision."""
    if df.empty or df['Confidence'].sum() == 0:
        return "HOLD", "Incomplete data from sub-modules. Standing by."
    
    avg_conf = df['Confidence'].mean()
    longs = len(df[df['Verdict'] == 'LONG'])
    shorts = len(df[df['Verdict'] == 'SHORT'])
    
    if avg_conf > 85 and longs > shorts:
        return "STRONG BUY", f"High confluence across {longs} models with {avg_conf:.1f}% confidence."
    elif avg_conf > 85 and shorts > longs:
        return "STRONG SELL", f"Heavy bearish alignment detected ({avg_conf:.1f}% confidence)."
    else:
        return "NEUTRAL", "Signals are divergent. Market regime is currently high-noise."

# --- UI LAYOUT ---
st.title("⚙️ Nexus Forge: Executive Decision Node")
target_asset = st.selectbox("🎯 Asset for Synthesis", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"])

if st.button("🛰️ Execute Global Synthesis"):
    with st.status("Gathering Intelligence...", expanded=True) as status:
        st.write("Polling Nexus Neural...")
        n_dir, n_conf, n_msg = get_module_recommendation("Nexus_Neural", target_asset)
        
        st.write("Polling Nexus Signal...")
        s_dir, s_conf, s_msg = get_module_recommendation("Nexus_Signal", target_asset)
        
        # Combine into DataFrame
        results_df = pd.DataFrame([
            {"Module": "Nexus Neural", "Verdict": n_dir, "Confidence": n_conf, "Status": n_msg},
            {"Module": "Nexus Signal", "Verdict": s_dir, "Confidence": s_conf, "Status": s_msg}
        ])
        status.update(label="Synthesis Complete!", state="complete", expanded=False)

    # 1. DISPLAY MASTER RECOMMENDATION
    st.write("---")
    final_action, explanation = generate_recommendation(results_df)
    
    # Large Recommendation Banner
    bg_color = "#00FFCC" if "BUY" in final_action else ("#FF4B4B" if "SELL" in final_action else "#333")
    st.markdown(f"""
        <div style="background-color:{bg_color}; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="color:black; margin:0;">EXECUTION: {final_action}</h1>
            <p style="color:black; font-weight:bold;">{explanation}</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. VISUALS
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Model Agreement")
        fig = go.Figure(go.Pie(labels=results_df['Module'], values=results_df['Confidence'], hole=.4, marker_colors=['#00FFCC', '#222']))
        fig.update_layout(template="plotly_dark", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("Sub-Module Metrics")
        st.dataframe(results_df, use_container_width=True)

st.caption(f"Nexus Forge v4.2 | Logic: Combined Hybrid | {time.strftime('%H:%M:%S')}")
