import os

# ============================================================
# DATABASE & FILES
# ============================================================

DB_FILE = os.getenv("DB_FILE", "nexus.db")
HISTORY_DB = os.getenv("HISTORY_DB", "nexus_history.db")
MODEL_FILE = os.getenv("MODEL_FILE", "nexus_brain.pkl")
BACKUP_DIR = os.getenv("BACKUP_DIR", "backups/")
PERFORMANCE_FILE = os.getenv("PERFORMANCE_FILE", "performance.json")

# ============================================================
# API & NOTIFICATIONS
# ============================================================

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ============================================================
# ASSET SETTINGS
# ============================================================

ASSETS = os.getenv(
    "ASSETS",
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,DOGE/USDT,XRP/USDT,ADA/USDT,PEPE/USDT,SUI/USDT,LTC/USDT"
).split(",")

DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "4h")
TIMEFRAMES = os.getenv("TIMEFRAMES", DEFAULT_TIMEFRAME).split(",")

# ============================================================
# ENGINE REGISTRY
# ============================================================

ENGINES = {
    "core": os.getenv("ENGINE_CORE", "Nexus Core"),
    "ai": os.getenv("ENGINE_AI", "Nexus AI (Phase 3 Ensemble)"),
    "hybrid_v1": os.getenv("ENGINE_HYBRID", "Nexus Hybrid"),
    "rangemaster": os.getenv("ENGINE_RANGEMASTER", "Nexus Rangemaster"),
}

# ============================================================
# PERFORMANCE & KILL-SWITCH SETTINGS
# ============================================================

KILL_THRESHOLD = float(os.getenv("KILL_THRESHOLD", 45.0))
RECOVERY_THRESHOLD = float(os.getenv("RECOVERY_THRESHOLD", 52.0))
MIN_TRADES_FOR_AUDIT = int(os.getenv("MIN_TRADES_FOR_AUDIT", 5))

# ============================================================
# PHASE 1: DYNAMIC RISK MANAGEMENT
# ============================================================

TOTAL_CAPITAL = float(os.getenv("TOTAL_CAPITAL", 1000.0))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", 2.0))
RR_RATIO = float(os.getenv("RR_RATIO", 2.0))
MIN_CONFIDENCE_FOR_SIZE_BOOST = float(os.getenv("MIN_CONFIDENCE_FOR_SIZE_BOOST", 75.0))

# --- SL/TP Specific ---
ATR_MULTIPLIER_SL = float(os.getenv("ATR_MULTIPLIER_SL", ATR_MULTIPLIER))
ATR_MULTIPLIER_TP = float(os.getenv("ATR_MULTIPLIER_TP", ATR_MULTIPLIER * RR_RATIO))

# ============================================================
# PHASE 2: GLOBAL REGIME FILTER
# ============================================================

BTC_CRASH_THRESHOLD = float(os.getenv("BTC_CRASH_THRESHOLD", -3.0))
GLOBAL_VOLATILITY_CAP = float(os.getenv("GLOBAL_VOLATILITY_CAP", 0.05))
REGIME_CHECK_INTERVAL = os.getenv("REGIME_CHECK_INTERVAL", "1h")

# ============================================================
# PHASE 3: ENSEMBLE SETTINGS
# ============================================================

VOTING_METHOD = os.getenv("VOTING_METHOD", "soft")
MIN_ENSEMBLE_CONFIDENCE = float(os.getenv("MIN_ENSEMBLE_CONFIDENCE", 55.0))

# ============================================================
# PHASE 4: MAINTENANCE SETTINGS
# ============================================================

MAX_SIGNAL_AGE_DAYS = int(os.getenv("MAX_SIGNAL_AGE_DAYS", 30))
PENDING_CLEANUP_HOURS = int(os.getenv("PENDING_CLEANUP_HOURS", 24))
BACKUP_COUNT = int(os.getenv("BACKUP_COUNT", 5))

# ============================================================
# FEATURE SETTINGS
# ============================================================

ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
EMA_PERIOD = int(os.getenv("EMA_PERIOD", 20))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
