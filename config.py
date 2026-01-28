import os

# --- DATABASE & FILES ---
DB_FILE = "nexus.db"
MODEL_FILE = "nexus_brain.pkl"  # Now contains the Ensemble object
PERFORMANCE_FILE = "performance.json"

# --- API & NOTIFICATIONS ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "YOUR_DISCORD_WEBHOOK_HERE")

# --- ASSET SETTINGS ---
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
DEFAULT_TIMEFRAME = "4h"

# --- ENGINE REGISTRY ---
ENGINES = {
    "core": "Nexus Core",
    "ai": "Nexus AI (Phase 3 Ensemble)",
    "hybrid_v1": "Nexus Hybrid",
    "rangemaster": "Nexus Rangemaster"
}

# --- PHASE 1, 2, 3 SETTINGS ---
RISK_PER_TRADE = 0.02         
TOTAL_CAPITAL = 1000.0        
ATR_MULTIPLIER = 2.0          
MIN_CONFIDENCE_FOR_SIZE_BOOST = 75.0  

# Guardian Settings
BTC_CRASH_THRESHOLD = -3.0    
GLOBAL_VOLATILITY_CAP = 0.05  

# Phase 3 Ensemble Settings
VOTING_METHOD = 'soft'  # 'soft' uses weighted probabilities; 'hard' uses majority vote
MIN_ENSEMBLE_CONFIDENCE = 55.0 # Only trade if the "Committee" is >55% sure
