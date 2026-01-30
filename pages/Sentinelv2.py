import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import sqlite3
import plotly.express as px

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="Aegis Sentinel Pro", layout="wide")
st.title("📡 Aegis Sentinel – Advanced Signal Engine + Full Backtesting")
st.caption("1H Entry • 4H/1D Confirmation • ML Confidence • TP/SL • Multi-Exchange Backtest & Analytics")

# ============================
# DATABASE SETUP
# ============================
conn = sqlite3.connect("signals.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS signals (
    timestamp TEXT,
    exchange TEXT,
    pair TEXT,
    direction TEXT,
    entry REAL,
    take_profit REAL,
    stop_loss REAL,
    confidence REAL,
    session TEXT
)
""")
conn.commit()

# ============================
# CONFIGURATION
# ============================
EXCHANGES = {
    "Bitget": ccxt.bitget(),
    "Gate.io": ccxt.gateio(),
    "XT": ccxt.xt()
}

ALL_SYMBOLS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","SUI/USDT","PEPE/USDT",
    "LINK/USDT","BNB/USDT","ADA/USDT","DOGE/USDT","MATIC/USDT"
]

st.sidebar.header("🔹 Trading Pair Selection")
selected_symbols = st.sidebar.multiselect(
    "Select assets to monitor", ALL_SYMBOLS, default=["BTC/USDT","ETH/USDT","SOL/USDT"]
)

st.sidebar.header("🔹 Strategy Parameters")
ema_fast_span = st.sidebar.slider("EMA Fast Span",5,50,20)
ema_slow_span = st.sidebar.slider("EMA Slow Span",10,200,50)
rsi_period = st.sidebar.slider("RSI Period",5,50,14)
atr_multiplier = st.sidebar.slider("ATR Multiplier (for TP/SL)",0.5,5.0,1.5, step=0.1)
confidence_threshold = st.sidebar.slider("ML Confidence Threshold",0.0,1.0,0.65)
bollinger_period = st.sidebar.slider("Bollinger Band Period",10,50,20)
bollinger_std = st.sidebar.slider("Bollinger Band Std Dev",1.0,3.0,2.0)
macd_fast = st.sidebar.slider("MACD Fast EMA",5,30,12)
macd_slow = st.sidebar.slider("MACD Slow EMA",10,60,26)
macd_signal = st.sidebar.slider("MACD Signal EMA",5,30,9)
risk_percent = st.sidebar.slider("Risk % per Trade (Backtest)", 0.5, 5.0, 1.0, step=0.5)

TIMEFRAMES = {"entry":"1h","confirm_4h":"4h","confirm_1d":"1d"}
MAX_BARS = 1000  # Longer history for meaningful backtesting

WEIGHTS = {"trend_alignment":0.3,"momentum":0.2,"volatility":0.2,"macd":0.2,"bollinger":0.1}

# ============================
# HARDEN CCXT
# ============================
for ex in EXCHANGES.values():
    ex.enableRateLimit = True
    ex.timeout = 30000

# ============================
# DATA FETCHING
# ============================
@st.cache_data(ttl=600)
def fetch_ohlcv(_exchange, symbol, timeframe):
    try:
        data = _exchange.fetch_ohlcv(symbol, timeframe, limit=MAX_BARS)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df
    except Exception as e:
        st.error(f"Fetch error {symbol} {timeframe}: {e}")
        return pd.DataFrame()

# ============================
# TECHNICAL INDICATORS (No look-ahead bias: adjust=False on EWMs)
# ============================
def compute_structure(df):
    if df.empty or len(df) < 50:
        return df
    
    # EMAs (causal, no look-ahead)
    df["ema_fast"] = df["close"].ewm(span=ema_fast_span, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_span, adjust=False).mean()
    
    # Proper Wilder RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    
    # Proper ATR (14-period)
    tr0 = df["high"] - df["low"]
    tr1 = (df["high"] - df["close"].shift()).abs()
    tr2 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # MACD (causal)
    ema_fast_macd = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow_macd = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["macd"] = ema_fast_macd - ema_slow_macd
    df["macd_signal"] = df["macd"].ewm(span=macd_signal, adjust=False).mean()
    
    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(bollinger_period).mean()
    df["bb_std"] = df["close"].rolling(bollinger_period).std()
    df["bb_upper"] = df["bb_mid"] + bollinger_std * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - bollinger_std * df["bb_std"]
    
    return df

def trend_bias(series_fast, series_slow):
    if series_fast > series_slow:
        return 1
    elif series_fast < series_slow:
        return -1
    return 0

def get_trading_session():
    utc_hour = datetime.utcnow().hour
    if 0 <= utc_hour < 8: return "Asian"
    elif 8 <= utc_hour < 16: return "London"
    else: return "NY"

# ============================
# ML CONFIDENCE
# ============================
def ml_confidence(row):
    # row contains current values for 1h, 4h, 1d trends
    trend_alignment = 1 if row["trend_alignment"] else 0
    momentum = abs(row["momentum_score"])
    volatility_score = row["volatility_score"]
    macd_sig = 1 if row["macd"] > row["macd_signal"] else 0
    bb_sig = 1 if row["close"] > row["bb_mid"] else 0
    
    raw = (WEIGHTS["trend_alignment"] * trend_alignment +
           WEIGHTS["momentum"] * momentum +
           WEIGHTS["volatility"] * volatility_score +
           WEIGHTS["macd"] * macd_sig +
           WEIGHTS["bollinger"] * bb_sig)
    return round(1 / (1 + np.exp(-5 * (raw - 0.5))), 4)

# ============================
# SIGNAL ENGINE
# ============================
def generate_signals(symbols):
    signals = []
    session = get_trading_session()
    for ex_name, ex in EXCHANGES.items():
        for symbol in symbols:
            try:
                df_1h = compute_structure(fetch_ohlcv(ex, symbol, "1h"))
                df_4h = compute_structure(fetch_ohlcv(ex, symbol, "4h"))
                df_1d = compute_structure(fetch_ohlcv(ex, symbol, "1d"))
                if df_1h.empty: continue
                
                # Get latest values
                latest = df_1h.iloc[-1]
                bias_1h = trend_bias(latest["ema_fast"], latest["ema_slow"])
                if bias_1h == 0: continue
                
                # Higher TF bias (last available)
                bias_4h = trend_bias(df_4h["ema_fast"].iloc[-1], df_4h["ema_slow"].iloc[-1]) if not df_4h.empty else bias_1h
                bias_1d = trend_bias(df_1d["ema_fast"].iloc[-1], df_1d["ema_slow"].iloc[-1]) if not df_1d.empty else bias_1h
                
                trends = [bias_1h, bias_4h, bias_1d]
                trend_alignment = len(set(trends)) == 1
                
                momentum_score = min(max((latest["rsi"] - 50)/50, -1), 1)
                vol = df_1h["close"].pct_change().std()
                volatility_score = 1 - min(vol * 10, 1) if not np.isnan(vol) else 0.5
                
                conf = ml_confidence({
                    "trend_alignment": trend_alignment,
                    "momentum_score": momentum_score,
                    "volatility_score": volatility_score,
                    "macd": latest["macd"],
                    "macd_signal": latest["macd_signal"],
                    "close": latest["close"],
                    "bb_mid": latest["bb_mid"]
                })
                if conf < confidence_threshold: continue
                
                atr = latest["atr"]
                if np.isnan(atr) or atr <= 0: continue
                
                entry_price = latest["close"]
                tp = entry_price + atr * atr_multiplier * bias_1h
                sl = entry_price - atr * bias_1h
                rr = round(atr_multiplier, 2)
                
                signal = {
                    "Exchange": ex_name,
                    "Pair": symbol,
                    "Direction": "LONG" if bias_1h == 1 else "SHORT",
                    "Entry": round(entry_price, 4),
                    "Take Profit": round(tp, 4),
                    "Stop Loss": round(sl, 4),
                    "Reward/Risk": rr,
                    "ML Confidence": conf,
                    "Session": session,
                    "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                }
                signals.append(signal)
                
                cursor.execute("""
                INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?)
                """, (signal["Time (UTC)"], signal["Exchange"], signal["Pair"], signal["Direction"],
                      signal["Entry"], signal["Take Profit"], signal["Stop Loss"], signal["ML Confidence"],
                      signal["Session"]))
                conn.commit()
            except Exception as e:
                st.warning(f"{ex_name} {symbol}: {str(e)}")
    return signals

# ============================
# BACKTEST ENGINE (Realistic, no look-ahead, per-exchange)
# ============================
def backtest(symbol):
    results = {}
    all_metrics = []
    reference_timestamps = None
    
    for ex_name, ex in EXCHANGES.items():
        try:
            df = compute_structure(fetch_ohlcv(ex, symbol, "1h"))
            df_4h = compute_structure(fetch_ohlcv(ex, symbol, "4h"))
            df_1d = compute_structure(fetch_ohlcv(ex, symbol, "1d"))
            if df.empty or len(df) < 200: 
                continue
            
            # Align higher TF trends
            df = df.copy()
            df["bias_4h"] = np.nan
            df["bias_1d"] = np.nan
            df = df.join(df_4h[["ema_fast", "ema_slow"]].rename(columns={"ema_fast":"ema_fast_4h", "ema_slow":"ema_slow_4h"}), how="left")
            df = df.join(df_1d[["ema_fast", "ema_slow"]].rename(columns={"ema_fast":"ema_fast_1d", "ema_slow":"ema_slow_1d"}), how="left")
            df[["ema_fast_4h","ema_slow_4h","ema_fast_1d","ema_slow_1d"]] = df[["ema_fast_4h","ema_slow_4h","ema_fast_1d","ema_slow_1d"]].ffill()
            df["bias_4h"] = np.sign(df["ema_fast_4h"] - df["ema_slow_4h"])
            df["bias_1d"] = np.sign(df["ema_fast_1d"] - df["ema_slow_1d"])
            
            equity = 10000.0
            equity_curve = [equity]
            peak = equity
            max_dd = 0.0
            position = 0  # 1 long, -1 short, 0 flat
            entry_price = 0.0
            sl = 0.0
            tp = 0.0
            size = 0.0
            trades = 0
            wins = 0
            gross_p = 0.0
            gross_l = 0.0
            
            for i in range(max(200, max(ema_slow_span, bollinger_period*2)), len(df)-1):  # Leave room for last bar
                row = df.iloc[i]
                current_price = row["close"]
                high = row["high"]
                low = row["low"]
                
                exited = False
                if position != 0:
                    # MTM equity
                    mtm_equity = equity + size * (current_price - entry_price) * position
                    equity_curve.append(mtm_equity)
                    if mtm_equity > peak:
                        peak = mtm_equity
                    dd = (peak - mtm_equity) / peak
                    max_dd = max(max_dd, dd)
                    
                    # Exit checks
                    if position == 1:
                        if low <= sl:
                            exit_price = sl
                            exited = True
                        elif high >= tp:
                            exit_price = tp
                            exited = True
                    else:
                        if high >= sl:
                            exit_price = sl
                            exited = True
                        elif low <= tp:
                            exit_price = tp
                            exited = True
                    
                    if exited:
                        pnl = size * (exit_price - entry_price) * position
                        equity += pnl
                        trades += 1
                        if pnl > 0:
                            wins += 1
                            gross_p += pnl
                        else:
                            gross_l += abs(pnl)
                        position = 0
                else:
                    equity_curve.append(equity)
                
                if position == 0:
                    # Signal check (entry next bar)
                    bias = trend_bias(row["ema_fast"], row["ema_slow"])
                    if bias == 0: continue
                    
                    trend_align = (bias == row["bias_4h"]) and (bias == row["bias_1d"])
                    momentum_score = min(max((row["rsi"] - 50)/50, -1), 1)
                    vol_score = 1 - min(df["close"].iloc[:i+1].pct_change().std() * 10, 1)
                    macd_sig = 1 if row["macd"] > row["macd_signal"] else 0
                    bb_sig = 1 if row["close"] > row["bb_mid"] else 0
                    
                    conf = ml_confidence({
                        "trend_alignment": trend_align,
                        "momentum_score": momentum_score,
                        "volatility_score": vol_score,
                        "macd": row["macd"],
                        "macd_signal": row["macd_signal"],
                        "close": row["close"],
                        "bb_mid": row["bb_mid"]
                    })
                    if conf < confidence_threshold: continue
                    
                    atr = row["atr"]
                    if np.isnan(atr) or atr <= 0: continue
                    
                    risk_amt = equity * (risk_percent / 100)
                    size = risk_amt / atr
                    entry_price = df.iloc[i+1]["open"]  # Realistic: enter at next bar open
                    sl = entry_price - atr * bias
                    tp = entry_price + atr * atr_multiplier * bias
                    position = bias
            
            # Final MTM or close
            final_price = df.iloc[-1]["close"]
            if position != 0:
                pnl = size * (final_price - entry_price) * position
                equity += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                    gross_p += pnl
                else:
                    gross_l += abs(pnl)
            equity_curve.append(equity)
            
            win_rate = round(wins / trades * 100, 1) if trades > 0 else 0
            pf = round(gross_p / gross_l, 2) if gross_l > 0 else ("∞" if gross_p > 0 else "0")
            total_ret = round((equity - 10000)/10000 * 100, 2)
            max_dd_pct = round(max_dd * 100, 2)
            
            all_metrics.append({
                "Exchange": ex_name,
                "Trades": trades,
                "Win Rate %": win_rate,
                "Profit Factor": pf,
                "Max DD %": max_dd_pct,
                "Total Return %": total_ret,
                "Final Equity": round(equity, 2)
            })
            results[ex_name] = equity_curve
            
            if reference_timestamps is None:
                reference_timestamps = df.index.tolist()
                
        except Exception as e:
            st.warning(f"Backtest error {ex_name} {symbol}: {e}")
    
    return results, all_metrics, reference_timestamps

# ============================
# STREAMLIT UI
# ============================
tabs = st.tabs(["📈 Live Signals", "📊 Backtesting", "📜 Audit Logs"])

with tabs[0]:
    st.subheader("Live Signals")
    if not selected_symbols:
        st.info("Please select trading pairs in the sidebar.")
    else:
        signals = generate_signals(selected_symbols)
        if signals:
            df_sig = pd.DataFrame(signals)
            
            def color_direction(row):
                color = "#d4edda" if row["Direction"] == "LONG" else "#f8d7da"
                return [f"background-color: {color}" for _ in row]
            
            def color_rr(val):
                if val >= 2.5:
                    return "background-color: #28a745; color: white"
                elif val >= 2.0:
                    return "background-color: #5cb85c; color: white"
                elif val >= 1.5:
                    return "background-color: #ffeeba; color: black"
                elif val >= 1.0:
                    return "background-color: #fff3cd; color: black"
                else:
                    return "background-color: #f5c6cb; color: black"
            
            styled = df_sig.style.apply(color_direction, axis=1)\
                                 .applymap(color_rr, subset=["Reward/Risk"])\
                                 .format({"ML Confidence": "{:.4f}", "Reward/Risk": "{:.2f}"})
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("No valid signals at this time.")

with tabs[1]:
    st.subheader("Backtesting & Multi-Exchange Comparison")
    if not selected_symbols:
        st.info("Please select trading pairs in the sidebar.")
    else:
        summary_metrics = []
        for symbol in selected_symbols:
            st.markdown(f"### {symbol}")
            equity_dict, metrics, timestamps = backtest(symbol)
            
            if metrics:
                metric_df = pd.DataFrame(metrics)
                st.dataframe(metric_df, use_container_width=True)
                
                # Per-pair aggregate (average across exchanges)
                avg_ret = metric_df["Total Return %"].mean()
                avg_dd = metric_df["Max DD %"].mean()
                total_trades = metric_df["Trades"].sum()
                summary_metrics.append({
                    "Pair": symbol,
                    "Avg Return %": round(avg_ret, 2),
                    "Avg Max DD %": round(avg_dd, 2),
                    "Total Trades": total_trades,
                    "Exchanges Tested": len(metrics)
                })
                
                if equity_dict and timestamps:
                    eq_df = pd.DataFrame(equity_dict, index=timestamps[:len(next(iter(equity_dict.values())))])
                    fig = px.line(eq_df, title=f"Equity Curves – {symbol} (Multi-Exchange Comparison)",
                                  labels={"value": "Equity ($)", "index": "Date"},
                                  height=500)
                    fig.update_layout(legend_title="Exchange")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for equity curve.")
            else:
                st.info(f"No backtest data for {symbol}")
        
        if summary_metrics:
            st.markdown("#### Overall Portfolio Summary (Across Selected Pairs)")
            summary_df = pd.DataFrame(summary_metrics)
            st.dataframe(summary_df, use_container_width=True)

with tabs[2]:
    st.subheader("Audit Logs")
    df_logs = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 200", conn)
    st.dataframe(df_logs, use_container_width=True)
