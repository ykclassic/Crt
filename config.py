import os

# --- DATABASE & FILES ---
DB_FILE = "nexus.db"
MODEL_FILE = "nexus_brain.pkl"
PERFORMANCE_FILE = "performance.json"

# --- API & NOTIFICATIONS ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "YOUR_DISCORD_WEBHOOK_HERE")

# --- ASSET SETTINGS ---
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
DEFAULT_TIMEFRAME = "4h"

# --- ENGINE REGISTRY ---
ENGINES = {
    "core": "Nexus Core",
    "ai": "Nexus AI (Phase 1)",
    "hybrid_v1": "Nexus Hybrid",
    "rangemaster": "Nexus Rangemaster"
}

# --- PERFORMANCE & KILL-SWITCH SETTINGS ---
KILL_THRESHOLD = 45.0      
RECOVERY_THRESHOLD = 52.0  
MIN_TRADES_FOR_AUDIT = 5   

# --- PHASE 1: DYNAMIC RISK MANAGEMENT ---
TOTAL_CAPITAL = 1000.0        
RISK_PER_TRADE = 0.02         
ATR_MULTIPLIER = 2.0          
RR_RATIO = 2.0                
MIN_CONFIDENCE_FOR_SIZE_BOOST = 75.0  

# --- PHASE 2: GLOBAL REGIME FILTER (CIRCUIT BREAKER) ---
BTC_CRASH_THRESHOLD = -3.0    # % drop in 1h to trigger Global Stop
GLOBAL_VOLATILITY_CAP = 0.05  # Max allowed ATR % before market is "Too Risky"
REGIME_CHECK_INTERVAL = "1h"

# --- FEATURE SETTINGS ---
ATR_PERIOD = 14               
EMA_PERIOD = 20               
RSI_PERIOD = 14               
