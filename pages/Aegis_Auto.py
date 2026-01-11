# ============================================================
# AEGIS INTELLIGENCE PRO v2
# Multi-Asset | Multi-Timeframe | Performance Analytics
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
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Aegis Intelligence Pro v2", layout="wide")

if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

# ============================================================
# TELEGRAM CONFIG
# ============================================================

TELEGRAM = {
    "enabled": True,
    "bot_token": "PUT_BOT_TOKEN",
    "chat_id": "PUT_CHAT_ID"
}

def send_telegram(msg):
    if not TELEGRAM["enabled"]:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM['bot_token']}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM["chat_id"], "text": msg}, timeout=5)
    except:
        pass

# ============================================================
# DATABASE
# ============================================================

conn = sqlite3.connect("aegis_signals.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS signals (
    timestamp TEXT,
    asset TEXT,
    timeframe TEXT,
    direction TEXT,
    entry REAL,
    stop REAL,
    target REAL,
    confidence REAL,
    outcome REAL
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

def features(df):
    df = df.copy()
    df["ema_9"] = df["c"].ewm(span=9).mean()
    df["ema_21"] = df["c"].ewm(span=21).mean()
    df["ema_50"] = df["c"].ewm(span=50).mean()
    df["vol_chg"] = df["v"].pct_change()
    df["atr"] = atr(df)
    df["target"] = df["c"].shift(-1)
    return df.dropna()

# ============================================================
# REGIME
# ============================================================

def regime(df):
    trend = abs(df["ema_9"].iloc[-1] - df["ema_50"].iloc[-1]) / df["c"].iloc[-1]
    vol = df["atr"].iloc[-1] / df["c"].iloc[-1]
    if trend > 0.01:
        return "TREND"
    if vol < 0.003:
        return "LOW_VOL"
    return "RANGE"

# ============================================================
# ENSEMBLE
# ============================================================

def ensemble_predict(df):
    feats = ["c","v","ema_9","ema_21","ema_50","vol_chg","atr"]
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[feats], train["target"]
    X_test, y_test = test[feats], test["target"]

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

    final_pred = np.dot(preds, weights)

    hit_rate = np.mean(
        np.sign(test["target"] - test["c"]) ==
        np.sign(final_pred - test["c"].iloc[-1])
    )

    return final_pred, min(99.0, hit_rate * 100)

# ============================================================
# TRADE ENGINE
# ============================================================

def trade_from_prediction(df, pred, conf, asset, tf):
    if conf < 60:
        return None

    price = df["c"].iloc[-1]
    atr_val = df["atr"].iloc[-1]
    direction = "LONG" if pred > price else "SHORT"

    stop = price - atr_val * 1.5 if direction == "LONG" else price + atr_val * 1.5
    target = price + atr_val * 3 if direction == "LONG" else price - atr_val * 3

    cursor.execute(
        "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,NULL)",
        (datetime.utcnow().isoformat(), asset, tf, direction, price, stop, target, conf)
    )
    conn.commit()

    return direction, price, stop, target

# ============================================================
# MULTI-ASSET SCANNER
# ============================================================

ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT"]
TIMEFRAMES = ["1h","4h"]

st.title("🧠 Aegis Intelligence Pro — Multi-Asset Scanner")

if st.button("🚀 Run Full Market Scan"):
    results = []

    for asset in ASSETS:
        df_1h = features(fetch_ohlcv(asset,"1h"))
        df_4h = features(fetch_ohlcv(asset,"4h"))

        pred1, conf1 = ensemble_predict(df_1h)
        pred4, conf4 = ensemble_predict(df_4h)

        dir1 = "LONG" if pred1 > df_1h["c"].iloc[-1] else "SHORT"
        dir4 = "LONG" if pred4 > df_4h["c"].iloc[-1] else "SHORT"

        if dir1 == dir4 and conf1 > 60 and conf4 > 60:
            trade = trade_from_prediction(df_1h, pred1, (conf1+conf4)/2, asset, "1h")
            if trade:
                results.append((asset, dir1, conf1, conf4))

    st.subheader("📊 Qualified Signals")
    st.dataframe(pd.DataFrame(results, columns=["Asset","Direction","Conf 1H","Conf 4H"]))

# ============================================================
# PERFORMANCE DASHBOARD
# ============================================================

st.write("---")
st.subheader("📈 Performance Dashboard")

df_perf = pd.read_sql("SELECT * FROM signals WHERE outcome IS NOT NULL", conn)

if not df_perf.empty:
    df_perf["equity"] = df_perf["outcome"].cumsum()
    drawdown = df_perf["equity"] - df_perf["equity"].cummax()

    st.metric("Total Trades", len(df_perf))
    st.metric("Win Rate", f"{(df_perf['outcome']>0).mean()*100:.2f}%")
    st.metric("Max Drawdown", f"{drawdown.min():.2f} R")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df_perf["equity"], name="Equity Curve"))
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No closed trades yet.")

# ============================================================
# FOOTER
# ============================================================

st.write("---")
st.caption("Walk-Forward Ensemble • Multi-TF Confluence • Risk-Adjusted Execution")
