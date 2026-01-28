import os

# --- DATABASE & FILES ---
DB_FILE = "nexus.db"
HISTORY_DB = "nexus_history.db"
MODEL_FILE = "nexus_brain.pkl"
BACKUP_DIR = "backups/"
PERFORMANCE_FILE = "performance.json"

# --- API & NOTIFICATIONS ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "YOUR_DISCORD_WEBHOOK_HERE")

# --- ASSET SETTINGS ---
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# --- PHASE 1, 2, 3, 4 SETTINGS ---
TOTAL_CAPITAL = 1000.0        
RISK_PER_TRADE = 0.02         
ATR_MULTIPLIER = 2.0          
MIN_CONFIDENCE_FOR_SIZE_BOOST = 75.0  

# Maintenance
MAX_SIGNAL_AGE_DAYS = 30      
PENDING_CLEANUP_HOURS = 24    
