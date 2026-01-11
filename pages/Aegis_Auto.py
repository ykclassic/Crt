# ============================================================
# AEGIS INTELLIGENCE — SIGNAL LAB
# Signal Generation • No Execution • Institutional Analytics
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import sqlite3
import requests

# ============================================================
# PAGE CONFIG & AUTH
# ============================================================

st.set_page_config(page_title="Aegis Intelligence | Signal Lab", layout="wide")

if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# ============================================================
# TELEGRAM (SIGNAL ALERTS ONLY)
# ============================================================

TELEGRAM = {
    "enabled": True,
    "bot_token": "PUT_BOT_TOKEN",
    "chat_id": "PUT_CHAT_ID"
}

def send_telegram(message: str):
    if not TELEGRAM["enabled"]:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM['bot_token']}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM["chat_id"], "text": message}, timeout=5)
    except:
        pass

# ============================================================
# SIGNAL DATABASE (NO EXECUTION DATA)
# ============================================================

conn = sqlite3.connect("aegis_signal_journal.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS signals (
    timestamp TEXT,
    asset TEXT,
    timeframe TEXT,
    bias TEXT,
    reference_price REAL,
    projected_objective REAL,
    invalidation_level REAL,
    confidence REAL,
    regime TEXT
)
""")
conn.commit()

# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, timeframe, limit=300):
    ex = ccxt.bitget()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","o","h","l","c","v"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def atr(df, period=14):
    tr = np.maximum(
        df["h"] - df["l"],
        np.maximum(abs(df["h"] - df["c"].shift()), abs(df["l"] - df["c"].shift()))
    )
    return tr.rolling(period).mean()

def engineer_features(df):
    df = df.copy()
    df["ema_9"] = df["c"].ewm(span=9).mean()
    df["ema_21"] = df["c"].ewm(span=21).mean()
    df["ema_50"] = df["c"].ewm(span=50).mean()
    df["vol_chg"] = df["v"].pct_change()
    df["atr"] = atr(df)
    df["target"] = df["c"].shift(-1)
    return df.dropna()

# ============================================================
# MARKET REGIME
# ============================================================

def detect_regime(df):
    trend = abs(df["ema_9"].iloc[-1] - df["ema_50"].iloc[-1]) / df["c"].iloc[-1]
    vol = df["atr"].iloc[-1] / df["c"].iloc[-1]
    if trend > 0.01:
        return "TRENDING"
    if vol < 0.003:
        return "LOW_VOLATILITY"
    return "RANGING"

# ============================================================
# WALK-FORWARD ENSEMBLE (SIGNAL-ONLY)
# ============================================================

def ensemble_signal(df):
    features = ["c","v","ema_9","ema_21","ema_50","vol_chg","atr"]
    split = int(len(df) * 0.8)

    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[features], train["target"]
    X_test, y_test = test[features], test["target"]

    models = [
        RandomForestRegressor(n_estimators=150, random_state=42),
        GradientBoostingRegressor(),
        Ridge(alpha=1.0)
    ]

    preds, errors = [], []

    for m in models:
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        preds.append(p[-1])
        errors.append(mean_absolute_error(y_test, m.predict(X_test)))

    weights = np.array([1/e for e in errors])
    weights /= weights.sum()

    final_prediction = np.dot(preds, weights)

    directional_accuracy = np.mean(
        np.sign(test["target"] - test["c"]) ==
        np.sign(final_prediction - test["c"].iloc[-1])
    )

    confidence = min(99.0, directional_accuracy * 100)

    return final_prediction, confidence

# ============================================================
# SIGNAL CONSTRUCTION (NO EXECUTION)
# ============================================================

def build_signal(df, prediction, confidence, asset, timeframe):
    if confidence < 60:
        return None

    price = df["c"].iloc[-1]
    atr_val = df["atr"].iloc[-1]
    bias = "LONG" if prediction > price else "SHORT"

    projected_objective = price + atr_val * 2 if bias == "LONG" else price - atr_val * 2
    invalidation = price - atr_val * 1.5 if bias == "LONG" else price + atr_val * 1.5

    reg = detect_regime(df)

    cursor.execute(
        "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?)",
        (
            datetime.utcnow().isoformat(),
            asset,
            timeframe,
            bias,
            price,
            projected_objective,
            invalidation,
            confidence,
            reg
        )
    )
    conn.commit()

    return {
        "asset": asset,
        "timeframe": timeframe,
        "bias": bias,
        "price": price,
        "objective": projected_objective,
        "invalidation": invalidation,
        "confidence": confidence,
        "regime": reg
    }

# ============================================================
# UI — MULTI-ASSET / MULTI-TF SCANNER
# ============================================================

ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT"]

st.title("🧠 Aegis Intelligence — Signal Lab")
st.caption("Signal Generation Only • No Execution • Institutional Analytics")

if st.button("🚀 Run Full Market Signal Scan"):
    signals = []

    for asset in ASSETS:
        df_1h = engineer_features(fetch_ohlcv(asset,"1h"))
        df_4h = engineer_features(fetch_ohlcv(asset,"4h"))

        pred_1h, conf_1h = ensemble_signal(df_1h)
        pred_4h, conf_4h = ensemble_signal(df_4h)

        dir_1h = "LONG" if pred_1h > df_1h["c"].iloc[-1] else "SHORT"
        dir_4h = "LONG" if pred_4h > df_4h["c"].iloc[-1] else "SHORT"

        if dir_1h == dir_4h and conf_1h > 60 and conf_4h > 60:
            signal = build_signal(
                df_1h,
                pred_1h,
                (conf_1h + conf_4h) / 2,
                asset,
                "1H (4H Confirmed)"
            )
            if signal:
                signals.append(signal)

                send_telegram(
                    f"""AEGIS SIGNAL (ANALYTICAL)
Asset: {signal['asset']}
Timeframe: {signal['timeframe']}
Bias: {signal['bias']}
Reference Price: {signal['price']:.2f}
Projected Objective: {signal['objective']:.2f}
Invalidation Level: {signal['invalidation']:.2f}
Confidence: {signal['confidence']:.2f}%
Regime: {signal['regime']}
"""
                )

    if signals:
        st.subheader("📊 Qualified Signals")
        st.dataframe(pd.DataFrame(signals))
    else:
        st.warning("No qualified signals under current market conditions.")

# ============================================================
# PERFORMANCE DASHBOARD (SIGNAL QUALITY)
# ============================================================

st.write("---")
st.subheader("📈 Signal Quality Dashboard")

df_log = pd.read_sql("SELECT * FROM signals", conn)

if not df_log.empty:
    st.metric("Total Signals Generated", len(df_log))
    st.metric("Average Confidence", f"{df_log['confidence'].mean():.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_log["confidence"], nbinsx=20))
    fig.update_layout(template="plotly_dark", height=300, title="Confidence Distribution")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No signals recorded yet.")

# ============================================================
# FOOTER
# ============================================================

st.write("---")
st.caption("Walk-Forward Ensemble • Multi-Timeframe Confluence • Signal-Only System")
