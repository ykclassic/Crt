# ============================================================
# AEGIS INTELLIGENCE — SIGNAL LAB v2
# Forward Validation • Regime Analytics • Calibration
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
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Aegis Signal Lab v2", layout="wide")

if "authenticated" not in st.session_state:
    st.switch_page("Home.py")
    st.stop()

FORWARD_BARS = 6  # validation window

# ============================================================
# DATABASE
# ============================================================

conn = sqlite3.connect("aegis_signal_lab.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS signals (
    timestamp TEXT,
    asset TEXT,
    timeframe TEXT,
    bias TEXT,
    ref_price REAL,
    objective REAL,
    invalidation REAL,
    confidence REAL,
    regime TEXT,
    validated INTEGER,
    correct INTEGER,
    bars_to_resolution INTEGER
)
""")
conn.commit()

# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300)
def fetch_ohlcv(symbol, timeframe, limit=400):
    ex = ccxt.bitget()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","o","h","l","c","v"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def atr(df, n=14):
    tr = np.maximum(
        df["h"] - df["l"],
        np.maximum(abs(df["h"] - df["c"].shift()), abs(df["l"] - df["c"].shift()))
    )
    return tr.rolling(n).mean()

def features(df):
    df = df.copy()
    df["ema9"] = df["c"].ewm(9).mean()
    df["ema21"] = df["c"].ewm(21).mean()
    df["ema50"] = df["c"].ewm(50).mean()
    df["atr"] = atr(df)
    df["target"] = df["c"].shift(-1)
    return df.dropna()

# ============================================================
# REGIME
# ============================================================

def regime(df):
    trend = abs(df["ema9"].iloc[-1] - df["ema50"].iloc[-1]) / df["c"].iloc[-1]
    vol = df["atr"].iloc[-1] / df["c"].iloc[-1]
    if trend > 0.01:
        return "TRENDING"
    if vol < 0.003:
        return "LOW_VOL"
    return "RANGING"

# ============================================================
# ENSEMBLE
# ============================================================

def ensemble(df):
    feats = ["c","v","ema9","ema21","ema50","atr"]
    split = int(len(df)*0.8)

    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[feats], train["target"]
    X_test, y_test = test[feats], test["target"]

    models = [
        RandomForestRegressor(150, random_state=42),
        GradientBoostingRegressor(),
        Ridge()
    ]

    preds, errors = [], []
    for m in models:
        m.fit(X_train, y_train)
        preds.append(m.predict(X_test)[-1])
        errors.append(mean_absolute_error(y_test, m.predict(X_test)))

    w = np.array([1/e for e in errors])
    w /= w.sum()

    pred = np.dot(preds, w)
    hit_rate = np.mean(
        np.sign(test["target"] - test["c"]) ==
        np.sign(pred - test["c"].iloc[-1])
    )

    return pred, min(99, hit_rate*100)

# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal(asset):
    df = features(fetch_ohlcv(asset, "1h"))
    pred, conf = ensemble(df)
    if conf < 60:
        return None

    price = df["c"].iloc[-1]
    bias = "LONG" if pred > price else "SHORT"
    atr_v = df["atr"].iloc[-1]

    obj = price + atr_v*2 if bias=="LONG" else price - atr_v*2
    inv = price - atr_v*1.5 if bias=="LONG" else price + atr_v*1.5

    cursor.execute("""
    INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,0,NULL,NULL)
    """, (
        datetime.utcnow().isoformat(),
        asset,
        "1H",
        bias,
        price,
        obj,
        inv,
        conf,
        regime(df)
    ))
    conn.commit()

    return asset, bias, conf

# ============================================================
# FORWARD VALIDATION ENGINE
# ============================================================

def validate_signals():
    df = pd.read_sql("SELECT rowid,* FROM signals WHERE validated=0", conn)
    for _, r in df.iterrows():
        candles = fetch_ohlcv(r.asset, "1h", FORWARD_BARS+5)
        candles = candles[candles["ts"] > pd.to_datetime(r.timestamp).value//10**6]
        resolved = False

        for i,row in candles.iterrows():
            if r.bias=="LONG":
                if row["h"] >= r.objective:
                    correct, bars = 1, i
                    resolved = True
                    break
                if row["l"] <= r.invalidation:
                    correct, bars = 0, i
                    resolved = True
                    break
            else:
                if row["l"] <= r.objective:
                    correct, bars = 1, i
                    resolved = True
                    break
                if row["h"] >= r.invalidation:
                    correct, bars = 0, i
                    resolved = True
                    break

        if resolved:
            cursor.execute("""
            UPDATE signals SET validated=1, correct=?, bars_to_resolution=?
            WHERE rowid=?
            """,(correct,bars,r.rowid))
            conn.commit()

# ============================================================
# UI
# ============================================================

ASSETS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT"]

st.title("🧠 Aegis Signal Lab — Research Edition")

if st.button("🚀 Generate Signals"):
    res = [generate_signal(a) for a in ASSETS]
    st.dataframe(pd.DataFrame([r for r in res if r], columns=["Asset","Bias","Confidence"]))

if st.button("🔍 Run Forward Validation"):
    validate_signals()
    st.success("Validation complete")

# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

st.write("---")
df = pd.read_sql("SELECT * FROM signals WHERE validated=1", conn)

if not df.empty:
    st.subheader("📊 Signal Analytics")

    # Regime accuracy
    st.write("### Regime Accuracy")
    st.dataframe(df.groupby("regime")["correct"].mean())

    # Signal decay
    st.write("### Signal Decay (Bars to Resolution)")
    st.bar_chart(df["bars_to_resolution"])

    # Confidence calibration
    df["bucket"] = pd.cut(df["confidence"], bins=[50,60,70,80,90,100])
    calib = df.groupby("bucket")["correct"].mean()
    st.write("### Confidence Calibration")
    st.line_chart(calib)

    # CSV export
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV Report", csv, "signals_report.csv")

    # PDF export
    if st.button("📄 Generate PDF Report"):
        pdf = SimpleDocTemplate("signal_report.pdf")
        styles = getSampleStyleSheet()
        content = [Paragraph("Aegis Signal Lab Report", styles["Title"])]
        content.append(Paragraph(df.describe().to_string(), styles["Normal"]))
        pdf.build(content)
        st.success("PDF generated (server-side)")
else:
    st.info("No validated signals yet.")

# ============================================================
# FOOTER
# ============================================================

st.write("---")
st.caption("Signal Science • No Execution • Institutional Research Framework")
