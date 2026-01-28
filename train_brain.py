import sqlite3
import pandas as pd
import numpy as np
import ccxt
import pickle
import logging
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from config import DB_FILE, MODEL_FILE

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

def train_model():
    logging.info("--- AI BRAIN TRAINING INITIALIZED ---")
    
    if not os.path.exists(DB_FILE):
        logging.error("No database found. AI cannot learn without history.")
        return

    # 1. Load historical signals from database
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT asset, timeframe, signal, entry, sl, tp, ts FROM signals"
    df_signals = pd.read_sql_query(query, conn)
    conn.close()

    if len(df_signals) < 10:
        logging.warning(f"Insufficient data ({len(df_signals)} signals). AI needs at least 10 historical trades to learn.")
        return

    # 2. Fetch market context for these signals to create training features
    # Using Gate.io for high-fidelity historical data
    ex = ccxt.gateio()
    training_data = []
    labels = []

    logging.info(f"Processing {len(df_signals)} signals for training...")

    for _, row in df_signals.iterrows():
        try:
            # Reconstruct the technical state at the time of the signal
            ts_ms = int(datetime.fromisoformat(row['ts']).timestamp() * 1000)
            # Fetch context (10 candles before the signal)
            ohlcv = ex.fetch_ohlcv(row['asset'], row['timeframe'], since=ts_ms - (10 * 60 * 60 * 1000), limit=20)
            if not ohlcv: continue
            
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # Feature Extraction (What the AI looks at)
            delta = df['c'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            rsi = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
            ema = df['c'].ewm(span=20).mean()
            dist_ema = (df['c'] - ema) / df['c']
            vol_change = df['v'].pct_change()

            # The Feature Vector
            feat = [rsi.iloc[-1], vol_change.iloc[-1], dist_ema.iloc[-1]]
            
            # The Label (Did it win or lose?)
            # We look at the next 50 candles to see if it hit TP before SL
            outcome_ohlcv = ex.fetch_ohlcv(row['asset'], row['timeframe'], since=ts_ms, limit=50)
            won = 0
            for candle in outcome_ohlcv:
                if row['signal'] == "LONG":
                    if candle[2] >= row['tp']: won = 1; break # High >= TP
                    if candle[3] <= row['sl']: won = 0; break # Low <= SL
                else:
                    if candle[3] <= row['tp']: won = 1; break # Low <= TP
                    if candle[2] >= row['sl']: won = 0; break # High >= SL
            
            if not np.isnan(feat).any():
                training_data.append(feat)
                labels.append(won)

        except Exception as e:
            continue

    if not training_data:
        logging.error("Could not reconstruct training features. Training aborted.")
        return

    # 3. Train the Model
    X = np.array(training_data)
    y = np.array(labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_scaled, y)
    
    accuracy = model.score(X_scaled, y)
    logging.info(f"Training Complete. Model Accuracy: {accuracy:.2%}")

    # 4. Save the "Brain"
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, scaler), f)
    
    logging.info(f"New brain saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
