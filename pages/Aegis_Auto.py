import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------
# 1️⃣ ATR Function
# -----------------------------
def atr(df, n=14):
    if df.empty:
        return pd.Series(dtype=float)
    tr = np.maximum(
        df["h"] - df["l"],
        np.maximum(abs(df["h"] - df["c"].shift()), abs(df["l"] - df["c"].shift()))
    )
    return tr.rolling(n).mean()

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------
def features(df):
    if df.empty:
        return df
    df = df.copy()
    df["ema9"] = df["c"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["c"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
    df["atr"] = atr(df)
    df["target"] = df["c"].shift(-1)
    return df.dropna()

# -----------------------------
# 3️⃣ Safe OHLCV Fetch
# -----------------------------
def safe_fetch(symbol, timeframe, limit=400):
    try:
        df = fetch_ohlcv(symbol, timeframe, limit)  # Your existing fetch_ohlcv function
        expected_cols = {"ts","o","h","l","c","v","dt"}
        if df.empty or not expected_cols.issubset(df.columns):
            st.warning(f"{symbol} {timeframe} OHLCV missing or incomplete")
            return pd.DataFrame(columns=list(expected_cols))
        return df
    except Exception as e:
        st.warning(f"{symbol} {timeframe} fetch failed: {e}")
        return pd.DataFrame(columns=["ts","o","h","l","c","v","dt"])

# -----------------------------
# 4️⃣ Multi-Asset Scan
# -----------------------------
ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT",
    "ADA/USDT","LINK/USDT","TRX/USDT","SUI/USDT","PEPE/USDT",
    "BNB/USDT","MATIC/USDT","LTC/USDT","AVAX/USDT","UNI/USDT",
    "ATOM/USDT","NEAR/USDT","FTM/USDT","ALGO/USDT","VET/USDT"
]

signals = []

for asset in ASSETS:
    reason = ""

    # Fetch and process features
    df_1h = features(safe_fetch(asset, "1h"))
    df_4h = features(safe_fetch(asset, "4h"))

    if df_1h.empty or df_4h.empty:
        st.info(f"{asset}: OHLCV data missing for 1H or 4H")
        continue

    # Ensemble predictions
    pred_1h, conf_1h = ensemble_signal(df_1h)
    pred_4h, conf_4h = ensemble_signal(df_4h)

    # Determine direction
    dir_1h = "LONG" if pred_1h > df_1h["c"].iloc[-1] else "SHORT"
    dir_4h = "LONG" if pred_4h > df_4h["c"].iloc[-1] else "SHORT"

    # Track filtering reason
    if conf_1h <= 60:
        reason = f"1H confidence too low ({conf_1h:.2f}%)"
    elif conf_4h <= 60:
        reason = f"4H confidence too low ({conf_4h:.2f}%)"
    elif dir_1h != dir_4h:
        reason = f"Direction mismatch (1H={dir_1h}, 4H={dir_4h})"

    # Generate signal if all conditions pass
    if not reason:
        signal = build_signal(
            df_1h,
            pred_1h,
            (conf_1h + conf_4h)/2,
            asset,
            "1H (4H Confirmed)"
        )
        if signal:
            signals.append(signal)
            # Optional: Telegram alert
            send_telegram(
                f"""AEGIS SIGNAL (ANALYTICAL)
Asset: {signal['asset']}
Timeframe: {signal['timeframe']}
Bias: {signal['bias']}
Reference Price: {signal['price']:.2f}
Projected Objective: {signal['objective']:.2f}
Invalidation Level: {signal['invalidation']:.2f}
Confidence: {signal['confidence']:.2f}%
Regime: {signal['regime']}"""
            )

    # Log reason if signal not generated
    if reason:
        st.info(f"{asset}: No signal generated → {reason}")
        # Optional DB logging
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals_log (
            timestamp TEXT,
            asset TEXT,
            reason TEXT
        )
        """)
        cursor.execute("""
        INSERT INTO signals_log (timestamp, asset, reason)
        VALUES (?,?,?)
        """, (datetime.utcnow().isoformat(), asset, reason))
        conn.commit()

# Display qualified signals
if signals:
    st.subheader("📊 Qualified Signals")
    st.dataframe(pd.DataFrame(signals))
else:
    st.warning("No qualified signals under current market conditions. Check reasons above.")
