import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import importlib.util
import os
import time

# 1. Dashboard Configuration
st.set_page_config(page_title="Nexus Forge | Executive Node", page_icon="⚙️", layout="wide")

# 2. Security Check (Inherited from Home.py)
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# --- THE AGGREGATION ENGINE ---
def poll_module(module_name, asset):
    """Dynamically imports and runs the core logic of a specific page."""
    path = f"pages/{module_name}.py"
    if not os.path.exists(path):
        return None
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        # Prevents the imported module from executing its own st.title/st.write
        spec.loader.exec_module(mod)
        
        if hasattr(mod, "get_live_signal"):
            return mod.get_live_signal(asset)
    except Exception as e:
        st.error(f"Error Polling {module_name}: {e}")
    return None

def compute_executive_decision(data):
    """
    Applies weights to module signals to generate a master recommendation.
    Weights: Neural (50%), Signal (30%), Profit (20%)
    """
    weights = {"Nexus_Neural": 0.50, "Nexus_Signal": 0.30, "Neural_Profit": 0.20}
    total_weighted_conf = 0
    votes = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
    
    for _, row in data.iterrows():
        w = weights.get(row['Module'], 0.10)
        total_weighted_conf += row['Confidence'] * w
        votes[row['Verdict']] += 1

    # Logic for Recommendation
    if total_weighted_conf > 85:
        verdict = "🚀 STRONG EXECUTION"
        color = "#00FFCC"
    elif total_weighted_conf > 70:
        verdict = "⚖️ ACCUMULATE"
        color = "#FFA500"
    else:
        verdict = "📡 STANDBY / NEUTRAL"
        color = "#808080"
        
    return verdict, round(total_weighted_conf, 2), color

# --- UI INTERFACE ---
st.title("⚙️ Nexus Forge: Executive Decision Node")
st.write("Centralized Multi-Agent Synthesis & Capital Allocation Engine")

target_asset = st.selectbox("🎯 Select Target for Synthesis", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"])

if st.button("🛰️ Initialize Global Aggregation"):
    with st.status("Polling Deep Learning & Signal Modules...", expanded=True) as status:
        modules = ["Nexus_Neural", "Nexus_Signal", "Neural_Profit"]
        results = []
        
        for m in modules:
            st.write(f"Syncing with {m}...")
            signal = poll_module(m, target_asset)
            if signal:
                direction, conf, _ = signal
                results.append({"Module": m, "Verdict": direction, "Confidence": conf})
            else:
                results.append({"Module": m, "Verdict": "OFFLINE", "Confidence": 0})
        
        df_results = pd.DataFrame(results)
        status.update(label="Synthesis Complete!", state="complete", expanded=False)

    # MASTER VERDICT DISPLAY
    decision, final_conf, theme_color = compute_executive_decision(df_results)
    
    st.markdown(f"""
        <div style="background-color:{theme_color}; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="color:black; margin:0;">{decision}</h1>
            <p style="color:black; font-weight:bold; font-size:1.2rem;">Confidence Score: {final_conf}%</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("---")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Model Consensus")
        # Visualizing the weight distribution
        fig = go.Figure(go.Bar(
            x=df_results['Module'], 
            y=df_results['Confidence'],
            marker_color=['#00FFCC', '#0099FF', '#FF00FF']
        ))
        fig.update_layout(template="plotly_dark", height=300, yaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Sub-Module Breakdown")
        st.dataframe(df_results, use_container_width=True)

st.caption(f"Nexus Forge v5.0 | Last Master Update: {time.strftime('%H:%M:%S')}")
