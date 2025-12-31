import streamlit as st
import ccxt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ============================
# CryptoForge Lite - Scikit-Learn Edition
# ============================
st.set_page_config(page_title="CryptoForge Lite", page_icon="🔮", layout="wide")
st.title("🔮 CryptoForge Lite")
st.markdown("**Rule-Based Signals + Machine Learning Price Prediction** (Lightweight • No TensorFlow)")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT']
    symbol = st.selectbox("Cryptocurrency", symbols, index=0)
    timeframe_options = ['15m', '1h', '4h', '1d']
    timeframe = st.selectbox("Timeframe", timeframe_options, index=2)
    prediction_horizon = st.selectbox("Predict Price In (Days)", [1, 7, 30], index=1)
    if st.button("Refresh Data"):
        st.session_state.clear()  # Force refresh

# ============================
# Data Fetching (Cached)
# ============================
@st.cache_data(ttl=300)  # Refresh every 5 minutes
def fetch_data(sym, tf):
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(sym, timeframe=tf, limit=2000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

df = fetch_data(symbol, timeframe)

if df.empty or len(df) < 100:
    st.error("Not enough data. Try a different symbol or timeframe.")
    st.stop()

# ============================
# Add Technical Indicators
# ============================
def add_indicators(data):
    df = data.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Trend
    df['sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['ema_20'] = ta.trend.ema_indicator(close, window=20)
    df['adx'] = ta.trend.ADXIndicator(high, low, close).adx()

    # Momentum
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['macd'] = ta.trend.MACD(close).macd()
    df['stoch_k'] = ta.momentum.StochasticOscillator(high, low, close).stoch()

    # Volatility
    bb = ta.volatility.BollingerBands(close)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_low'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    # Volume
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    return df.dropna()

df_ind = add_indicators(df)

# ============================
# Rule-Based Signal
# ============================
def generate_signal(row):
    buy_score = 0
    sell_score = 0

    # Price vs moving averages
    if row['close'] > row['sma_20']: buy_score += 1
    else: sell_score += 1
    if row['close'] > row['ema_20']: buy_score += 1
    else: sell_score += 1

    # RSI
    if row['rsi'] < 30: buy_score += 2
    if row['rsi'] > 70: sell_score += 2

    # MACD
    if row['macd'] > 0: buy_score += 1
    else: sell_score += 1

    # Bollinger Bands
    if row['close'] < row['bb_low']: buy_score += 2
    if row['close'] > row['bb_high']: sell_score += 2

    # ADX (trend strength)
    if row['adx'] > 25: buy_score += 1

    # Final decision
    if buy_score >= sell_score + 2:
        return "STRONG BUY", "green"
    elif buy_score > sell_score:
        return "BUY", "lightgreen"
    elif sell_score >= buy_score + 2:
        return "STRONG SELL", "red"
    elif sell_score > buy_score:
        return "SELL", "pink"
    else:
        return "NEUTRAL", "gray"

latest = df_ind.iloc[-1]
signal_text, signal_color = generate_signal(latest)

# ============================
# Layout
# ============================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Live Market Signal")
    st.metric("Current Price", f"${latest['close']:,.2f}")
    st.markdown(f"<h2 style='color:{signal_color}; text-align:center;'>{signal_text}</h2>", unsafe_allow_html=True)

    st.subheader("Key Indicators")
    st.write(f"RSI: {latest['rsi']:.1f}")
    st.write(f"MACD: {latest['macd']:.4f}")
    st.write(f"ADX (Trend Strength): {latest['adx']:.1f}")
    st.write(f"ATR (Volatility): ${latest['atr']:.2f}")

with col2:
    st.subheader(f"Price Chart - {symbol} ({timeframe})")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_ind.index, open=df_ind['open'], high=df_ind['high'],
                                 low=df_ind['low'], close=df_ind['close'], name="Price"))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['sma_20'], name="SMA 20", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['bb_high'], name="BB High", line=dict(dash="dash", color="red")))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['bb_low'], name="BB Low", line=dict(dash="dash", color="green")))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================
# ML Prediction with RandomForest
# ============================
st.subheader("🤖 Machine Learning Price Prediction (Random Forest)")

feature_cols = ['close', 'rsi', 'macd', 'sma_20', 'ema_20', 'atr', 'bb_high', 'bb_low', 'adx', 'obv', 'stoch_k']
target_col = f'close_shift_{-prediction_horizon}'

# Prepare target: future price
df_ml = df_ind.copy()
df_ml[target_col] = df_ml['close'].shift(-prediction_horizon)

# Drop rows where future price is unknown (end of data)
df_ml = df_ml.dropna(subset=[target_col])

X = df_ml[feature_cols]
y = df_ml[target_col]

if len(X) < 50:
    st.warning("Not enough historical data for reliable prediction yet.")
else:
    # Train model on all but last row
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X[:-1], y[:-1])

    # Predict next horizon
    current_features = X.iloc[-1:].values
    predicted_price = model.predict(current_features)[0]

    delta = predicted_price - latest['close']
    pct_change = (delta / latest['close']) * 100

    st.metric(
        label=f"Predicted Price in {prediction_horizon} days",
        value=f"${predicted_price:,.2f}",
        delta=f"{delta:+,.2f} ({pct_change:+.2f}%)"
    )

    if pct_change > 5:
        st.success("🚀 Model expects strong upward move")
    elif pct_change > 0:
        st.info("↗️ Mild bullish outlook")
    elif pct_change > -5:
        st.info("↘️ Mild bearish outlook")
    else:
        st.warning("📉 Model expects strong downward move")

st.caption("CryptoForge Lite • Powered by scikit-learn • Fast • Reliable • December 27, 2025")