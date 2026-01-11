import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

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
# 3️⃣ Fetch OHLCV
# -----------------------------
def fetch_ohlcv(symbol, timeframe='1h', limit=200):
    try:
        ex = ccxt.bitget()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
        df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.warning(f"{symbol} {timeframe} fetch failed: {e}")
        return pd.DataFrame(columns=['ts','o','h','l','c','v','dt'])

# -----------------------------
# 4️⃣ Safe Fetch Wrapper
# -----------------------------
def safe_fetch(symbol, timeframe='1h', limit=200):
    df = fetch_ohlcv(symbol, timeframe, limit)
    expected_cols = {"ts","o","h","l","c","v","dt"}
    if df.empty or not expected_cols.issubset(df.columns):
        st.info(f"{symbol} {timeframe}: OHLCV missing or incomplete")
        return pd.DataFrame(columns=list(expected_cols))
    return df

# -----------------------------
# 5️⃣ Ensemble Signal (RF + EMA)
# -----------------------------
def ensemble_signal(df):
    if df.empty:
        return 0, 0
    df_train = df.dropna()
    feature_cols = ['c', 'v', 'ema9', 'ema21', 'atr']
    if df_train.empty or any(col not in df_train.columns for col in feature_cols):
        return 0, 0
    X = df_train[feature_cols]
    y = df_train['target']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    last_features = X.iloc[-1].values.reshape(1, -1)
    pred_price = model.predict(last_features)[0]
    # Confidence based on EMA spread
    ema_diff = abs(df['ema9'].iloc[-1] - df['ema21'].iloc[-1]) / df['ema21'].iloc[-1]
    confidence = min(98.5, 75 + (ema_diff * 500))
    return pred_price, confidence

# -----------------------------
# 6️⃣ Build Signal
# -----------------------------
def build_signal(df, pred_price, conf, asset, timeframe_label):
    if df.empty:
        return None
    return {
        "asset": asset,
        "timeframe": timeframe_label,
        "bias": "LONG" if pred_price > df["c"].iloc[-1] else "SHORT",
        "price": df["c"].iloc[-1],
        "objective": pred_price,
        "invalidation": df["c"].iloc[-1] * 0.995,
        "confidence": conf,
        "regime": "RF+EMA Ensemble"
    }

# -----------------------------
# 7️⃣ Assets
# -----------------------------
ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT",
    "ADA/USDT","LINK/USDT","TRX/USDT","SUI/USDT","PEPE/USDT",
    "BNB/USDT","MATIC/USDT","LTC/USDT","AVAX/USDT","UNI/USDT",
    "ATOM/USDT","NEAR/USDT","FTM/USDT","ALGO/USDT","VET/USDT"
]

# -----------------------------
# 8️⃣ SUMMARY DASHBOARD
# -----------------------------
summary = []

for asset in ASSETS:
    df_1h = features(safe_fetch(asset, "1h"))
    df_4h = features(safe_fetch(asset, "4h"))

    if df_1h.empty or df_4h.empty:
        summary.append({"asset": asset, "status": "No Signal", "reason": "Missing OHLCV"})
        continue

    pred_1h, conf_1h = ensemble_signal(df_1h)
    pred_4h, conf_4h = ensemble_signal(df_4h)
    dir_1h = "LONG" if pred_1h > df_1h["c"].iloc[-1] else "SHORT"
    dir_4h = "LONG" if pred_4h > df_4h["c"].iloc[-1] else "SHORT"

    reason = ""
    if conf_1h <= 60:
        reason = f"1H confidence too low ({conf_1h:.2f}%)"
    elif conf_4h <= 60:
        reason = f"4H confidence too low ({conf_4h:.2f}%)"
    elif dir_1h != dir_4h:
        reason = f"Direction mismatch (1H={dir_1h},4H={dir_4h})"

    status = "Signal Generated" if not reason else "No Signal"
    summary.append({"asset": asset, "status": status, "reason": reason if reason else "-"})

summary_df = pd.DataFrame(summary)
st.subheader("📊 Multi-Asset Signal Summary")
st.dataframe(summary_df)
st.markdown("**❗ Filters Blocking Signals:**")
st.write(summary_df['reason'].value_counts())

# -----------------------------
# 9️⃣ Generate Signals & Charts
# -----------------------------
signals = []

for asset in ASSETS:
    df_1h = features(safe_fetch(asset, "1h"))
    df_4h = features(safe_fetch(asset, "4h"))

    if df_1h.empty or df_4h.empty:
        continue

    pred_1h, conf_1h = ensemble_signal(df_1h)
    pred_4h, conf_4h = ensemble_signal(df_4h)
    dir_1h = "LONG" if pred_1h > df_1h["c"].iloc[-1] else "SHORT"
    dir_4h = "LONG" if pred_4h > df_4h["c"].iloc[-1] else "SHORT"

    reason = ""
    if conf_1h <= 60:
        reason = f"1H confidence too low ({conf_1h:.2f}%)"
    elif conf_4h <= 60:
        reason = f"4H confidence too low ({conf_4h:.2f}%)"
    elif dir_1h != dir_4h:
        reason = f"Direction mismatch (1H={dir_1h},4H={dir_4h})"

    if reason:
        st.info(f"{asset}: No signal generated → {reason}")
        continue

    # Build qualified signal
    signal = build_signal(df_1h, pred_1h, (conf_1h+conf_4h)/2, asset, "1H (4H Confirmed)")
    signals.append(signal)

    # -----------------------------
    # Plot Chart for the Signal
    # -----------------------------
    current_price = df_1h["c"].iloc[-1]
    future_dt = [df_1h["dt"].iloc[-1] + timedelta(hours=i) for i in range(5)]
    future_prices = [current_price + ((pred_1h - current_price)/4)*i for i in range(5)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_1h["dt"].tail(30), y=df_1h["c"].tail(30),
                             name="Actual Price", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=future_dt, y=future_prices,
                             name="Predicted Move", line=dict(color="orange", dash="dash")))
    fig.update_layout(title=f"{asset} Signal Chart | Bias: {signal['bias']} | Confidence: {signal['confidence']:.2f}%",
                      template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Display all qualified signals in a table
if signals:
    st.subheader("📈 Qualified Signals Table")
    st.dataframe(pd.DataFrame(signals))
else:
    st.warning("No qualified signals under current market conditions. Check reasons above.")
