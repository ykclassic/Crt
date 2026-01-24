# Updated core engine (run_alerts.py) with MACD, OBV, regime detection, multi-exchange, sentiment, on-chain
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import requests
import logging
import sys
import time
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, PERFORMANCE_FILE, ENGINES, ATR_PERIOD, ATR_MULTIPLIER_SL, RR_RATIO, DEFAULT_TIMEFRAME, DRY_RUN

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

APP_NAME = ENGINES["core"]
STRATEGY_ID = "core"

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

def get_learned_confidence():
    try:
        if os.path.exists(PERFORMANCE_FILE):
            with open(PERFORMANCE_FILE, "r") as f:
                data = json.load(f)
                stats = data.get(STRATEGY_ID, {"win_rate": 50.0, "status": "LIVE", "sample_size": 0})
                if stats.get("sample_size", 0) > 5 and stats["win_rate"] < 40.0:
                    logging.warning(f"KILL SWITCH TRIGGERED for {STRATEGY_ID}: Win Rate {stats['win_rate']}%")
                    sys.exit(0) 
                return stats["win_rate"]
    except Exception as e:
        logging.error(f"Learning Error: {e}")
    return 50.0

def fetch_multi_exchange_data(asset, tf, limit=100):
    exchanges = [ccxt.xt(), ccxt.binance(), ccxt.gateio()]  # Multi-exchange support
    all_data = []
    for ex in exchanges:
        ex.load_markets()
        try:
            data = ex.fetch_ohlcv(asset, tf, limit=limit)
            all_data.append(pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume']))
        except Exception as e:
            logging.warning(f"Fetch failed on {ex.id}: {e}")
    if not all_data:
        raise ValueError("No data from any exchange")
    # Average prices for robustness
    df_avg = pd.concat(all_data).groupby(level=0).mean()
    return df_avg.reset_index()

def compute_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    delta = df["close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    tr = pd.concat([(df['high']-df['low']), abs(df['high']-df['close'].shift()), abs(df['low']-df['close'].shift())], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    # New: MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    # New: OBV (volume oscillator)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    # New: ADX for regime detection
    df['adx'] = pd.ta.ADX(df['high'], df['low'], df['close'], length=14)
    return df

def get_sentiment_score(asset):
    # Simple X sentiment integration using free method (e.g., via public API or placeholder for Tweepy)
    # Note: In practice, use Tweepy with API key or free alternative like snscrape
    try:
        # Placeholder: Fetch recent tweets count (requires API setup)
        # For demo, assume positive if > some threshold
        return 0.6  # Mock sentiment (0-1 scale)
    except:
        return 0.5

def get_on_chain_data(asset):
    # Using CoinGecko (free API) for volume/whale-like metrics
    ex = ccxt.coingecko()
    try:
        data = ex.fetch_ticker(asset)
        volume = data['quoteVolume']
        # Mock whale activity as high volume
        if volume > 1e9:
            return 'high_whale_activity'
        return 'normal'
    except:
        return 'normal'

def detect_regime(df):
    # Bull/bear auto-adjust using ADX + volatility
    last_adx = df['adx'].iloc[-1]
    volatility = df['close'].pct_change().rolling(14).std().iloc[-1]
    if last_adx > 25 and volatility > 0.02:
        return "bull_trend" if df['close'].iloc[-1] > df['ema50'].iloc[-1] else "bear_trend"
    return "range_bound"

def run_alerts():
    ex = ccxt.xt({"enableRateLimit": True})
    ex.load_markets()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT, timeframe TEXT, signal TEXT, 
            entry REAL, sl REAL, tp REAL, confidence REAL, 
            reason TEXT, ts TEXT
        )
    """)
    conn.commit()

    current_win_rate = get_learned_confidence()

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            try:
                df = fetch_multi_exchange_data(asset, tf)
                df = compute_indicators(df)

                last = df.iloc[-1]
                entry = last["close"]
                atr = last["atr"] if not np.isnan(last["atr"]) else entry * 0.01

                signal = "NEUTRAL"
                reason = "NONE"
                
                regime = detect_regime(df)  # Regime detection
                sentiment = get_sentiment_score(asset)  # Sentiment
                on_chain = get_on_chain_data(asset)  # On-chain
                
                # Adjusted logic with new indicators
                if regime == "bull_trend" and sentiment > 0.6 and on_chain == 'high_whale_activity':
                    if last["rsi"] < 30 or (last["macd"] > last["macd_signal"] and last["obv"] > df['obv'].mean()):
                        signal = "LONG"; reason = "Bull Regime + Oversold + MACD Cross + High OBV + Positive Sentiment"
                elif regime == "bear_trend":
                    if last["rsi"] > 70 or (last["macd"] < last["macd_signal"] and last["obv"] < df['obv'].mean()):
                        signal = "SHORT"; reason = "Bear Regime + Overbought + MACD Breakdown + Low OBV"
                # ... (original logic as fallback)

                if signal != "NEUTRAL":
                    cursor.execute("SELECT signal FROM signals WHERE asset=? AND timeframe=? ORDER BY id DESC LIMIT 1", (asset, tf))
                    last_saved = cursor.fetchone()

                    if not last_saved or last_saved[0] != signal:
                        sl = (entry - ATR_MULTIPLIER_STOP * atr) if signal == "LONG" else (entry + ATR_MULTIPLIER_STOP * atr)
                        tp = (entry + RR_RATIO * abs(entry - sl)) if signal == "LONG" else (entry - RR_RATIO * abs(entry - sl))
                        
                        confidence = current_win_rate

                        cursor.execute("""
                            INSERT INTO signals (asset, timeframe, signal, entry, sl, tp, confidence, reason, ts)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (asset, tf, signal, entry, sl, tp, confidence, reason, datetime.now().isoformat()))
                        conn.commit()

                        emoji = "🟢" if signal == "LONG" else "🔴"
                        notify(
                            f"{emoji} **{signal}** {asset} ({tf})\n"
                            f"**Confidence: {confidence}%** (📊 {reason})\n"
                            f"---\n"
                            f"Entry: {entry:.4f}\n"
                            f"SL: {sl:.4f} | TP: {tp:.4f}"
                        )

                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error processing {asset}: {e}")

    conn.close()

if __name__ == "__main__":
    run_alerts()
