import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Nexus AI Brain Inspector", layout="wide")

st.title("🧠 Nexus AI: Brain Visualization")
st.write("This tool interprets the binary `.pkl` file to show you exactly how the AI makes decisions.")

if not os.path.exists("nexus_brain.pkl"):
    st.error("Model file 'nexus_brain.pkl' not found. Please run training first.")
else:
    with open("nexus_brain.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    # 1. Feature Importance Section
    st.header("🎯 What is the AI looking at?")
    features = ["RSI (Momentum)", "Volume Change (Volatility)", "EMA Distance (Trend)"]
    importances = model.feature_importances_
    
    feat_df = pd.DataFrame({'Metric': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.bar(feat_df, x='Importance', y='Metric', orientation='h', 
                     title="AI Priority Weighting", color='Importance',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Natural Language Interpretation")
        top_feat = feat_df.iloc[0]['Metric']
        st.write(f"The AI is currently **heavily biased towards {top_feat}**.")
        st.write("""
        * **High Importance**: This metric is the strongest predictor of whether a trade will be a WIN or a LOSS.
        * **Low Importance**: The AI has learned to ignore this metric as it currently represents 'noise' in the Bitget/Gate.io markets.
        """)

    st.divider()

    # 2. Strategy Simulator
    st.header("🧪 Logic Simulator")
    st.write("Move the sliders to see how the AI reacts to different market conditions.")
    
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        sim_rsi = st.slider("Current RSI", 0, 100, 50)
    with s_col2:
        sim_vol = st.slider("Volume Change (%)", -1.0, 2.0, 0.0)
    with s_col3:
        sim_ema = st.slider("Distance from EMA (%)", -0.10, 0.10, 0.0)

    # Prepare data for prediction
    test_feat = np.array([[sim_rsi, sim_vol, sim_ema]])
    test_scaled = scaler.transform(test_feat)
    prob = model.predict_proba(test_scaled)[0][1] * 100

    st.metric(label="AI Confidence in a LONG position", value=f"{prob:.2f}%")
    
    if prob > 60:
        st.success("🤖 AI Verdict: **Strong Long Sentiment**. The pattern matches historical wins.")
    elif prob < 40:
        st.error("🤖 AI Verdict: **Strong Short Sentiment**. The pattern matches historical losses.")
    else:
        st.warning("🤖 AI Verdict: **Neutral/Uncertain**. Historical data for this pattern is mixed.")
