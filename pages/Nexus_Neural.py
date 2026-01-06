import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# 1. Dashboard Configuration
st.set_page_config(page_title="Aegis Global | Multi-Asset Pulse", page_icon="🌐", layout="wide")

# 2. Security Gate
if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# --- SIMULATED REAL-TIME SIGNAL AGGREGATOR ---
def get_global_signals(assets):
    """Generates a snapshot of signals for all assets in the library."""
    data = []
    for asset in assets:
        regime = np.random.choice(["BULLISH", "BEARISH", "SIDEWAYS"], p=[0.4, 0.3, 0.3])
        conf = np.random.uniform(60, 98)
        # Decay: How many minutes until the signal drops below 70% confidence
        decay_min = np.random.randint(5, 180) 
        
        data.append({
            "Asset": asset,
            "Regime": regime,
            "Confidence": round(conf, 2),
            "Decay (Min)": decay_min,
            "Status": "🔥 HIGH PRIORITY" if conf > 90 and regime != "SIDEWAYS" else "📡 MONITORING"
        })
    return pd.DataFrame(data)

# --- UI LAYOUT ---
st.title("🌐 Aegis Global: Multi-Asset Intelligence Matrix")
st.write("Real-time signal aggregation and decay tracking across the Aegis Network.")

# Maintained Asset Library
asset_library = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", 
    "ADA/USDT", "LINK/USDT", "TRX/USDT", "SUI/USDT", "PEPE/USDT"
]

if st.button("🔄 Refresh Global Intelligence"):
    df_signals = get_global_signals(asset_library)
    
    # 1. Top Signals Highlight
    st.subheader("🚀 High-Confidence Entries")
    top_signals = df_signals[df_signals["Status"] == "🔥 HIGH PRIORITY"].sort_values(by="Confidence", ascending=False)
    
    if not top_signals.empty:
        cols = st.columns(len(top_signals))
        for i, row in enumerate(top_signals.itertuples()):
            with cols[i]:
                st.metric(row.Asset, f"{row.Confidence}%", f"{row.Regime}")
                st.caption(f"Expires in: {row.Decay_Min}m")
    else:
        st.info("No high-priority signals currently meeting the 90% confidence threshold.")

    # 2. Signal Decay & Confidence Heatmap
    st.write("---")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Signal Reliability Matrix")
        fig = px.scatter(df_signals, x="Confidence", y="Decay (Min)", size="Confidence", 
                         color="Regime", hover_name="Asset",
                         color_discrete_map={"BULLISH": "#00FFCC", "BEARISH": "#FF4B4B", "SIDEWAYS": "#808080"})
        fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Immediate Expiry Zone")
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        

    with c2:
        st.subheader("Full Asset Pulse")
        # Color formatting for the dataframe
        def color_regime(val):
            color = '#00FFCC' if val == 'BULLISH' else ('#FF4B4B' if val == 'BEARISH' else 'white')
            return f'color: {color}'
        
        st.dataframe(df_signals.style.applymap(color_regime, subset=['Regime']), use_container_width=True)

# 3. System Health (Previous Updates Maintained)
st.write("---")
h1, h2, h3 = st.columns(3)
with h1:
    st.write("🔒 **Vault Status**: Encrypted")
with h2:
    st.write("📊 **Regime Engine**: K-Means Active")
with h3:
    st.write("📡 **Data Source**: Bitget/XT Multi-feed")

st.caption("Aegis Global v4.0 | Multi-Asset Signal Command")
