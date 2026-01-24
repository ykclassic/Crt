# 🛡️ Nexus Intelligence Suite
**The Self-Evolving Quantitative Trading Framework**

Nexus is a multi-engine intelligence suite designed to scan crypto markets, generate high-probability signals via a consensus of "specialist" bots, and self-train a Deep Neural Network to filter market noise.



---

## 🧠 The Intelligence Architecture
The suite is composed of four distinct strategy engines that feed into a central consensus judge:

1.  **Nexus Core:** Mean Reversion specialist (EMA20/RSI).
2.  **Hybrid V1:** Trend Momentum tracker (SMA50).
3.  **Rangemaster:** Volatility & Range specialist (Bollinger Bands).
4.  **AI Predict:** Deep Learning pattern recognition (MLP Classifier).

---

## 📊 Statistical Auditing & Trust Score
Nexus does not rely on static logic. Every trade is audited in real-time to calculate a **Trust Score**. This prevents "lucky streaks" from being mistaken for profitable strategies.

### The Trust Formula
We use a weighted significance formula to determine which engine is currently the most reliable:

$$Trust = \frac{WinRate \times \sqrt{SampleSize}}{10}$$

* **Win Rate:** The percentage of successful trades based on TP/SL hits.
* **Sample Size:** The number of trades recorded. The square root ensures that as the sample grows, the trust score becomes more stable.
* **Kill Switch:** Any engine falling below a **40% Win Rate** is automatically moved to **RECOVERY** mode and silenced until performance improves.

---

## 🤖 Self-Training AI (Deep Network)
The `train_brain.py` script executes every hour via GitHub Actions to prevent "Model Drift."

* **Model:** Multi-Layer Perceptron (MLP) Neural Network.
* **Training Data:** The most recent 500 hours of market price action and volatility metrics.
* **Persistence:** The model is saved as `nexus_brain.pkl` and pushed back to the repository automatically.



---

## 🚀 Quick Start Guide

### 1. GitHub Deployment
1.  **Fork this Repository** to your own account.
2.  **Add Secrets:** Go to `Settings > Secrets and Variables > Actions` and add:
    * `DISCORD_WEBHOOK_URL`: Your Discord channel webhook.
3.  **Enable Permissions:** Go to `Settings > Actions > General` and set **Workflow permissions** to "Read and write permissions."

### 2. Dashboard Access
The dashboard can be run locally or hosted on **Streamlit Cloud**:
```bash
pip install streamlit plotly pandas
streamlit run dashboard.py
