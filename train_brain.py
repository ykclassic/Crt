import pandas as pd
import numpy as np
import ccxt
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def train_brain():
    print("📡 Fetching historical data for training...")
    ex = ccxt.xt()
    
    # 1. Fetch 500 hours of data for BTC
    ohlcv = ex.fetch_ohlcv("BTC/USDT", '1h', limit=500)
    df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])

    # 2. Feature Engineering
    # RSI Calculation
    delta = df['c'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    df['rsi'] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean()))
    
    # Volume Change
    df['vol_change'] = df['v'].pct_change()
    
    # Distance from EMA20
    df['ema20'] = df['c'].ewm(span=20).mean()
    df['dist_ema'] = (df['c'] - df['ema20']) / df['c']
    
    # 3. Labeling (The Target)
    # Did the price go UP in the next 4 hours? (1 = Yes, 0 = No)
    df['target'] = (df['c'].shift(-4) > df['c']).astype(int)
    
    # 4. Cleanup
    df = df.dropna()
    X = df[['rsi', 'vol_change', 'dist_ema']]
    y = df['target']

    # 5. Scaling & Training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Deep Network: 2 layers (64 nodes and 32 nodes)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    # 6. Save both the Model and the Scaler
    with open("nexus_brain.pkl", "wb") as f:
        pickle.dump((model, scaler), f)
    
    print("🧠 Deep Network trained successfully and saved as nexus_brain.pkl")

if __name__ == "__main__":
    train_brain()
