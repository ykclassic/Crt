import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Nexus AI Brain Inspector", layout="wide", page_icon="🧠")

st.title("🧠 Nexus AI: Brain Visualization")
st.write("Interpreting the `.pkl` file to reveal the AI's current decision-making logic.")

if not os.path.exists("nexus_brain.pkl"):
    st.error("Model file 'nexus_brain.pkl' not found. Please run training first.")
else:
    try:
        with open("nexus_brain.pkl", "rb") as f:
            # Unpack the tuple
            data = pickle.load(f)
            
            # Resilience check: Ensure we handle both (model, scaler) and raw model
            if isinstance(data, tuple):
                model, scaler = data
            else:
                model = data
                scaler = None

        # 1. Feature Importance Section
        st.header("🎯 What is the AI looking at?")
        
        # Check if model has the attribute before accessing
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Index must match train_brain.py: [RSI, Vol_Change, Dist_EMA]
            features = ["RSI (Momentum)", "Volume Change (Volatility)", "EMA Distance (Trend)"]
            
            feat_df = pd.DataFrame({'Metric': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

            col1, col2 = st.columns([1, 1])

            with col1:
                fig = px.bar(feat_df, x='Importance', y='Metric', orientation='h', 
                             title="AI Priority Weighting", 
                             color='Importance',
                             color_continuous_scale='Bluered')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Natural Language Interpretation")
                top_feat = feat_df.iloc[0]['Metric']
                st.info(f"The AI is currently **heavily biased towards {top_feat}**.")
                st.write(f"""
                - **Primary Driver**: {top_feat} is determining {feat_df.iloc[0]['Importance']:.1%} of the signal confidence.
                - **Learning State**: The AI has found that this specific metric currently filters out the most noise on Gate.io/XT.
                """)
        else:
            st.warning("The loaded model does not support feature importance visualization yet. Complete more training cycles.")

        st.divider()

        # 2. Logic Simulator
        if scaler:
            st.header("🧪 Logic Simulator")
            st.write("Simulate market conditions to see the AI's predicted win probability.")
            
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                sim_rsi = st.slider("Current RSI", 0, 100, 50)
            with s_col2:
                sim_vol = st.slider("Volume Change (%)", -1.0, 2.0, 0.0)
            with s_col3:
                sim_ema = st.slider("Distance from EMA (%)", -0.10, 0.10, 0.0)

            # Prepare and Scale data
            test_feat = np.array([[sim_rsi, sim_vol, sim_ema]])
            test_scaled = scaler.transform(test_feat)
            
            # Get Prediction Probability
            prob = model.predict_proba(test_scaled)[0][1] * 100

            st.metric(label="AI Confidence in a LONG position", value=f"{prob:.2f}%")
            
            if prob > 60:
                st.success("🤖 AI Verdict: **Strong Long Sentiment**. Pattern matches historical wins.")
            elif prob < 40:
                st.error("🤖 AI Verdict: **Strong Short Sentiment**. Pattern matches historical losses.")
            else:
                st.warning("🤖 AI Verdict: **Neutral/Uncertain**. Market noise is too high for this pattern.")

    except Exception as e:
        st.error(f"Error interpreting the brain: {e}")
        st.info("Check if train_brain.py has successfully saved a model after at least 10 signals.")
