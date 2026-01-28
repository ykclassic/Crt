import os

# --- DATABASE & FILES ---
DB_FILE = "nexus.db"
HISTORY_DB = "nexus_history.db"
MODEL_FILE = "nexus_brain.pkl"
BACKUP_DIR = "backups/"
PERFORMANCE_FILE = "performance.json"

# --- API & NOTIFICATIONS ---
# In GitHub Actions, ensure you have a Secret named WEBHOOK_URL
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

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

# --- PHASE 2: GLOBAL REGIME FILTER (MARKET GUARDIAN) ---
# THESE ARE THE MISSING VARIABLES CAUSING YOUR ERROR
BTC_CRASH_THRESHOLD = -3.0    # % drop in 1h to trigger Global Stop
GLOBAL_VOLATILITY_CAP = 0.05  # Max allowed ATR % before market is "Too Risky"
REGIME_CHECK_INTERVAL = "1h"

# --- PHASE 3: ENSEMBLE SETTINGS ---
VOTING_METHOD = 'soft'  
MIN_ENSEMBLE_CONFIDENCE = 55.0 

# --- PHASE 4: MAINTENANCE SETTINGS ---
MAX_SIGNAL_AGE_DAYS = 30      
PENDING_CLEANUP_HOURS = 24    
BACKUP_COUNT = 5              

# --- FEATURE SETTINGS ---
ATR_PERIOD = 14               
EMA_PERIOD = 20               
RSI_PERIOD = 14               
