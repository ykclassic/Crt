# ============================================================
# AEGIS INTELLIGENCE PRO
# Institutional-Grade AI Trading Intelligence System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import requests
import sqlite3
import math

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aegis Intelligence Pro",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# SECURITY GATE
# ============================================================

if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# ============================================================
# TELEGRAM CONFIG (REQUIRED)
# ============================================================

TELEGRAM_CONFIG = {
    "enabled": True,
    "bot_token": "PUT_YOUR_BOT_TOKEN_HERE",
    "chat_id": "PUT_YOUR_CHAT_ID_HERE"
}

# ============================================================
# DATABASE (SIGNAL AUDIT TRAIL)
# ============================================================

conn = sqlite3.connect("aegis_signals.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS signals (
    timestamp TEXT,
    asset TEXT,
    direction TEXT,
    entry REAL,
    stop REAL,
    target REAL,
    confidence REAL,
    regime TEXT
)
""")
conn.commit()

# ============================================================
# UTILITIES
# ============================================================

def send_telegram(msg: str):
    if not TELEGRAM_CONFIG["enabled"]:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_CONFIG['bot_token']}/sendMessage"
        payload = {"chat_id": TELEGRAM_CONFIG["chat_id"], "text": msg}
        requests.post(url, json=payload, timeout=5)
    except:
        pass


def atr(df, period=14):
    tr = np.maximum(
        df['h'] - df['l'],
        np.maximum(
            abs(df['h'] - df['c'].shift()),
            abs(df['l'] - df['c'].shift())
        )
    )
    return tr.rolling(period).mean()


# ============================================================
# DATA FETCH (CACHED)
# ============================================================

@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, timeframe="1h", limit=300):
    ex = ccxt.bitget()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):
    df = df.copy()
    df["ema_9"] = df["c"].ewm(span=9).mean()
    df["ema_21"] = df["c"].ewm(span=21).mean()
    df["ema_50"] = df["c"].ewm(span=50).mean()
    df["vol_chg"] = df["v"].pct_change()
    df["atr"] = atr(df)
    df["target"] = df["c"].shift(-1)
    return df.dropna()


# ============================================================
# REGIME DETECTION
# ============================================================

def detect_regime(df):
    vol = df["atr"].iloc[-1] / df["c"].iloc[-1]
    trend = abs(df["ema_9"].iloc[-1] - df["ema_50"].iloc[-1]) / df["c"].iloc[-1]

    if vol < 0.003:
        return "LOW_VOL"
    if trend > 0.01:
        return "TRENDING"
    return "RANGING"


# ============================================================
# WALK-FORWARD ENSEMBLE MODEL
# ============================================================

def walk_forward_ensemble(df):
    features = ["c","v","ema_9","ema_21","ema_50","vol_chg","atr"]
    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    models = {
        "rf": RandomForestRegressor(n_estimators=150, random_state=42),
        "gb": GradientBoostingRegressor(),
        "ridge": Ridge(alpha=1.0)
    }

    preds = []
    errors = []

    for m in models.values():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        preds.append(p[-1])
        errors.append(mean_absolute_error(y_test, m.predict(X_test)))

    weights = np.array([1/e for e in errors])
    weights /= weights.sum()

    final_pred = np.dot(preds, weights)

    directional_hits = np.mean(
        np.sign(test["target"] - test["c"]) ==
        np.sign(np.array(preds).mean() - test["c"].iloc[-1])
    )

    confidence = min(99.0, directional_hits * 100)

    return final_pred, confidence


# ============================================================
# EXECUTION ENGINE
# ============================================================

def build_trade(df, prediction, confidence, asset):
    price = df["c"].iloc[-1]
    atr_val = df["atr"].iloc[-1]

    direction = "NO TRADE"
    if confidence > 60:
        direction = "LONG" if prediction > price else "SHORT"

    if direction == "NO TRADE":
        return None

    risk_pct = 0.01
    stop_dist = atr_val * 1.5
    target_dist = stop_dist * 2

    entry = price
    stop = entry - stop_dist if direction == "LONG" else entry + stop_dist
    target = entry + target_dist if direction == "LONG" else entry - target_dist

    cursor.execute(
        "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(),
            asset,
            direction,
            entry,
            stop,
            target,
            confidence,
            detect_regime(df)
        )
    )
    conn.commit()

    return {
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "target": target,
        "confidence": confidence
    }


# ============================================================
# UI
# ============================================================

st.title("🧠 Aegis Intelligence Pro")
st.caption("Institutional-Grade AI Trading System")

asset = st.selectbox(
    "Select Asset",
    ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT"]
)

if st.button("🚀 Run AI Scan"):
    df_raw = fetch_ohlcv(asset)
    df = build_features(df_raw)

    regime = detect_regime(df)
    pred, conf = walk_forward_ensemble(df)
    trade = build_trade(df, pred, conf, asset)

    st.subheader("📊 AI Verdict")
    st.metric("Confidence", f"{conf:.2f}%")
    st.info(f"Market Regime: {regime}")

    if trade:
        st.success(trade["direction"])
        st.write(trade)

        send_telegram(
            f"""
AEGIS SIGNAL
Asset: {asset}
Direction: {trade['direction']}
Entry: {trade['entry']:.2f}
Stop: {trade['stop']:.2f}
Target: {trade['target']:.2f}
Confidence: {trade['confidence']:.2f}%
"""
        )
    else:
        st.warning("No Trade — Insufficient Edge")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["dt"], y=df["c"], name="Price"))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.write("---")
st.info("Model: Walk-Forward Ensemble | Execution: Risk-Adjusted | Alerts: Telegram")
