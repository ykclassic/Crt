import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import importlib.util
import os
import time

# 1. Page Configuration
st.set_page_config(page_title="Nexus Forge | Aggregator", page_icon="⚙️", layout="wide")

# 2. Dynamic Module Importer
def import_page_module(page_name):
    """Dynamically imports functions from other Streamlit pages."""
    path = f"pages/{page_name}.py"
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(page_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# --- AGGREGATION ENGINE ---
def run_global_aggregation():
    """Triggers logic across all key modules and collects results."""
    results = []
    
    # Modules to poll
    modules_to_poll = {
        "Nexus_Neural": "run_neural_inference",
        "Nexus_Signal": "get_confluence_signal", # Assumes this function exists in Nexus_Signal
    }
    
    for page, func_name in modules_to_poll.items():
        mod = import_page_module(page)
        if mod and hasattr(mod, func_name):
            # Run the specific inference function from that page
            # We pass a default asset like 'BTC/USDT'
            direction, conf = getattr(mod, func_name)("BTC/USDT")
            results.append({"Module": page, "Direction": direction, "Confidence": conf})
        else:
            results.append({"Module": page, "Direction": "N/A", "Confidence": 0})
            
    return pd.DataFrame(results)

# --- UI LAYOUT ---
st.title("⚙️ Nexus Forge: Master Aggregator")
st.write("Aggregating live inference from all Aegis sub-modules.")

if st.button("🛰️ Poll All Modules & Synthesize"):
    with st.spinner("Synchronizing with Neural and Signal nodes..."):
        df_master = run_global_aggregation()
        
        # Calculate Informed Decision (Weighted Average)
        avg_conf = df_master["Confidence"].mean()
        
        # Layout
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Aggregated Verdict")
            st.metric("Global Confidence", f"{avg_conf:.1f}%")
            if avg_conf > 85:
                st.success("🔥 HIGH CONVICTION: Multi-model alignment detected.")
            else:
                st.warning("⚖️ DIVERGENCE: Models are not in sync.")
        
        with c2:
            st.subheader("Module Breakdown")
            st.table(df_master)

        # Signal Consensus Visualization
        
        fig = go.Figure(data=go.Scatterpolar(
          r=df_master['Confidence'],
          theta=df_master['Module'],
          fill='toself',
          marker=dict(color='#00FFCC')
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                          template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

st.caption("Nexus Forge v2.0 | Pull-based Architecture | Live Sync")
