import ccxt
import pandas as pd
import numpy as np
import os  # For env vars
import sqlite3
import requests
import logging
import pickle
import tweepy  # For X sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # VADER for better slang handling
from datetime import datetime
from config import DB_FILE, WEBHOOK_URL, MODEL_FILE, PERFORMANCE_FILE, ENGINES, ATR_PERIOD, ATR_MULTIPLIER_SL, RR_RATIO, DEFAULT_TIMEFRAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

APP_NAME = ENGINES["ai"]
STRATEGY_ID = "ai"

def notify(msg):
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"content": f"**[{APP_NAME}]**\n{msg}"})
        except Exception as e:
            logging.error(f"Discord notify failed: {e}")

def get_ai_prediction(rsi, price, ema):
    try:
        if not os.path.exists(MODEL_FILE): return None
        with open(MODEL_FILE, "rb") as f: model, scaler = pickle.load(f)
        dist_ema = (price - ema) / price
        feat = np.array([[rsi, 0.0, dist_ema]]) # vol_change set to 0 for simplicity
        feat_scaled = scaler.transform(feat)
        return round(model.predict_proba(feat_scaled)[0][1] * 100, 2)
    except: return None

def get_sentiment_score(asset):
    # Setup API (keys from env vars)
    auth = tweepy.OAuth1UserHandler(
        consumer_key=os.getenv("M5QEboUAgw8R8A6wnPYNuXykT"),
        consumer_secret=os.getenv("BjK5pjhNNhwX9j2wd3xxmTlaAaZapH7Z40Ry9j2MRhgBQKcoz9"),
        access_token=os.getenv("1915433899-jSHoatAiI1ZyihLra1MhXBnuZvlZU3T7FIgJwJS"),
        access_token_secret=os.getenv("D5myVYz0aFhuVTq1UVzjsBhuux3gj6reuCS732IG06VIe")
    )
    api = tweepy.API(auth)
    
    try:
        tweets = api.search_tweets(q=asset + " sentiment", count=100, lang="en")
        # Analyze sentiment using VADER (better for crypto slang like "moon", "rekt")
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(tweet.text)['compound'] for tweet in tweets]
        avg_score = sum(scores) / len(scores) if scores else 0.5
        return (avg_score + 1) / 2  # Normalize to 0-1
    except Exception as e:
        logging.error(f"Sentiment fetch error: {e}")
        return 0.5

def run_ai_logic():
    ex = ccxt.xt({"enableRateLimit": True})
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT, signal TEXT, entry REAL, sl REAL, tp REAL, 
            confidence REAL, reason TEXT, ts TEXT
        )
    """)
    conn.commit()

    # Learning Check
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, "r") as f: perf = json.load(f).get(STRATEGY_ID, {"status": "LIVE"})
        if perf["status"] == "RECOVERY": return

    for asset in ["BTC/USDT", "ETH/USDT"]:
        try:
            data = ex.fetch_ohlcv(asset, '1h', limit=100)
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v'])
            df['ema20'] = df['c'].rolling(20).mean()
            
            # RSI
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            df['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            
            last = df.iloc[-1]
            ai_conf = get_ai_prediction(last['rsi'], last['c'], last['ema20'])
            
            # Integrate sentiment
            sentiment = get_sentiment_score(asset)
            reason = "DEEP NETWORK" if ai_conf else "HEURISTIC FALLBACK"
            reason += f" | Sentiment: {sentiment:.2f}"
            
            final_conf = ai_conf if ai_conf else 52.0
            # Adjust confidence based on sentiment
            final_conf = final_conf * (1 + (sentiment - 0.5) * 0.2)  # +/- 10% boost based on sentiment
            
            signal = "LONG" if final_conf > 50 else "SHORT"
            
            price = last['c']
            sl = price * (0.98 if signal == "LONG" else 1.02)
            tp = price * (1.04 if signal == "LONG" else 0.96)

            cursor.execute("""
                INSERT INTO signals (asset, signal, entry, sl, tp, confidence, reason, ts) 
                VALUES (?,?,?,?,?,?,?,?)""",
                (asset, signal, price, sl, tp, final_conf, reason, datetime.now().isoformat()))
            conn.commit()
            
            notify(f"🤖 **AI Prediction**\nAsset: {asset}\nSignal: {signal}\n**Confidence: {final_conf}%** (📊 {reason})")
        except Exception as e:
            print(f"Error: {e}")
    conn.close()

if __name__ == "__main__":
    run_ai_logic()
