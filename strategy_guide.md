# 📘 Nexus Intelligence Suite: Strategy Reference Guide

This guide defines the mathematical "personality" of each engine and the specific market conditions required for a high-conviction trade.

---

### 1. Nexus Core (The Anchor)
* **Role:** Mean Reversion Specialist.
* **Primary Logic:** Uses **EMA 20** and **RSI**. It identifies when the price has "overextended" (stretched) too far from its average.
* **Ideal Market:** Range-bound or trending with frequent pullbacks.
* **Entry Trigger:** Price rejects the EMA 20 while RSI shows a return from extreme levels.
* **Weakness:** Parabolic moves (vertical "moons" or "crashes") where price refuses to return to the mean.

### 2. Hybrid V1 (The Trend Follower)
* **Role:** Momentum Tracker.
* **Primary Logic:** Uses **SMA 50** crossovers. It acts as the "Big Wave" filter to ensure you aren't fighting the primary market direction.
* **Ideal Market:** Strong, sustained Bull or Bear trends.
* **Entry Trigger:** Price crosses and holds above (Long) or below (Short) the 50-period moving average.
* **Weakness:** "Chop" or "Sawtooth" markets where price fluctuates sideways over the moving average.

### 3. Rangemaster (The Volatility Specialist)
* **Role:** Boundary Trader.
* **Primary Logic:** Uses **Bollinger Bands (2.0 SD)**. It defines the statistical "ceiling" and "floor" of current price action.
* **Ideal Market:** Low to medium volatility; Sideways/Consolidating markets.
* **Entry Trigger:** Price touches the Upper Band (Short) or Lower Band (Long) and begins to curl back.
* **Weakness:** Volatility breakouts. If a "squeeze" releases, price can ride the bands for a long time, leading to premature entries.

### 4. AI Predict (The Pattern Filter)
* **Role:** Deep Learning Gatekeeper.
* **Primary Logic:** **MLP Neural Network**. It analyzes the hidden correlation between RSI, Volume changes, and Distance-from-Mean.
* **Ideal Market:** Any market with high historical data volume.
* **Entry Trigger:** Probability score > 50% based on the last 500 hours of similar market patterns.
* **Weakness:** "Black Swan" events or news-driven spikes that deviate from historical technical behavior.

---

## 💎 The Consensus Hierarchy

| Tier | Name | Requirement | Confidence Level |
| :--- | :--- | :--- | :--- |
| **Tier 1** | 💎 **Diamond** | **4 Engines** Agree | **Institutional Grade:** Maximum Confluence. |
| **Tier 2** | 🥇 **Gold** | **3 Engines** Agree | **High Conviction:** High probability setup. |
| **Tier 3** | 🥈 **Silver** | **2 Engines** Agree | **Standard:** Routine technical setup. |
| **Tier 4** | 👤 **Solo** | **1 Engine** Alone | **Speculative:** High risk/Learning phase. |

---

## 🛡️ Operational Risk Rules

1.  **Trust Score Threshold:** Do not trade an engine manually until its **Trust Score > 40** on the Dashboard.
2.  **Recovery Protocol:** If an engine status is **RECOVERY**, ignore its individual signals. It is currently being "retrained" by the Audit Intelligence script.
3.  **The Confluence Rule:** A Diamond Consensus (4 engines) effectively cancels out the individual weaknesses of each engine (e.g., Hybrid's chop weakness is filtered by Rangemaster's sideways strength).

---
*Last Updated: January 2026*
