import os

# --- DATABASE & FILES ---
DB_FILE = "nexus.db"
HISTORY_DB = "nexus_history.db" # New for Phase 4
MODEL_FILE = "nexus_brain.pkl"
BACKUP_DIR = "backups/"        # New for Phase 4
PERFORMANCE_FILE = "performance.json"

# --- API & NOTIFICATIONS ---
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "YOUR_DISCORD_WEBHOOK_HERE")

# --- ASSET SETTINGS ---
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# --- ENGINE REGISTRY ---
ENGINES = {
    "core": "Nexus Core",
    "ai": "Nexus AI (Phase 3 Ensemble)",
    "hybrid_v1": "Nexus Hybrid",
    "rangemaster": "Nexus Rangemaster"
}

# --- PERFORMANCE & RISK ---
TOTAL_CAPITAL = 1000.0        
RISK_PER_TRADE = 0.02         
ATR_MULTIPLIER = 2.0          
MIN_CONFIDENCE_FOR_SIZE_BOOST = 75.0  

# --- PHASE 4: MAINTENANCE SETTINGS ---
MAX_SIGNAL_AGE_DAYS = 30      # Move to history after 30 days
PENDING_CLEANUP_HOURS = 24    # Delete untriggered 1h signals after 24h
BACKUP_COUNT = 5              # Keep the last 5 brain versions
