import streamlit as st
import ccxt
import pandas as pd
import plotly.express as px
import ta
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# App config
st.set_page_config(page_title="CryptoForge ML", layout="wide")
st.title('🔮 CryptoForge ML - Signals + LSTM Prediction')

symbols = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT']
timeframe_options = ['5m', '15m', '1h', '4h', '1d']

# Sidebar
symbol = st.sidebar.selectbox('Cryptocurrency', symbols)
timeframe = st.sidebar.selectbox('Timeframe', timeframe_options)
horizon_days = st.sidebar.selectbox('Prediction Horizon (days)', [1, 7, 30])

@st.cache_data(ttl=300)
def fetch_data(symbol, timeframe, limit=2000):
    exchange = ccxt.binance({'enableRateLimit': True})
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return df

df = fetch_data(symbol, timeframe)

if df.empty or len(df) < 100:
    st.error("Insufficient data fetched.")
    st.stop()

# Indicators
def add_indicators(df):
    df['ma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['bb_high'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
    df['bb_low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    return df

df = add_indicators(df)

# Simple signal
def generate_signal(row):
    buy_score = sell_score = 0
    if row['close'] > row['ma_20']: buy_score += 1
    else: sell_score += 1
    if row['rsi'] < 30: buy_score += 2
    if row['rsi'] > 70: sell_score += 2
    if row['macd'] > 0: buy_score += 1
    else: sell_score += 1
    if row['close'] < row['bb_low']: buy_score += 1
    if row['close'] > row['bb_high']: sell_score += 1
    if row['adx'] > 25: buy_score += 1

    if buy_score > sell_score + 1: return "STRONG BUY"
    elif buy_score > sell_score: return "BUY"
    elif sell_score > buy_score + 1: return "STRONG SELL"
    elif sell_score > buy_score: return "SELL"
    else: return "NEUTRAL"

latest = df.iloc[-1]
signal = generate_signal(latest)

# Display
col1, col2 = st.columns(2)
with col1:
    st.subheader('Current Price & Signal')
    st.metric("Price", f"${latest['close']:,.2f}")
    color = "green" if "BUY" in signal else "red" if "SELL" in signal else "gray"
    st.markdown(f"<h2 style='color:{color}; text-align:center;'>{signal}</h2>", unsafe_allow_html=True)

with col2:
    st.subheader('Price Chart')
    fig = px.line(df, y='close', title=f'{symbol} - {timeframe}')
    fig.add_scatter(y=df['ma_20'], name='MA20')
    fig.add_scatter(y=df['bb_high'], name='BB High')
    fig.add_scatter(y=df['bb_low'], name='BB Low')
    st.plotly_chart(fig, use_container_width=True)

# LSTM Prediction
st.subheader('LSTM Price Prediction')

@st.cache_resource
def train_lstm_model(_df):
    data = _df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
    return model, scaler

model, scaler = train_lstm_model(df)

if st.button('Predict Future Price'):
    last_seq = scaler.transform(df['close'][-60:].values.reshape(-1, 1))
    pred_scaled = model.predict(last_seq.reshape(1, 60, 1), verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    st.success(f"Predicted {symbol} price in {horizon_days} days: **${pred_price:,.2f}**")
    st.info("Note: LSTM predictions on raw prices are highly uncertain in crypto markets.")