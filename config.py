import os

# =========================
# Exchange
# =========================
EXCHANGE_ID = "xt"  # Unified routing target

# =========================
# Assets
# =========================
ASSETS = os.getenv(
    "ASSETS",
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT"
).split(",")

TRADING_PAIRS = ASSETS

# =========================
# Timeframes
# =========================
EXECUTION_TF = "1h"
CONFIRM_TF_1 = "4h"
CONFIRM_TF_2 = "1d"

TIMEFRAMES = [EXECUTION_TF, CONFIRM_TF_1, CONFIRM_TF_2]

# =========================
# Risk Model
# =========================
RISK_PERCENT = 0.02
REWARD_PERCENT = 0.05

# =========================
# Database
# =========================
DB_FILE = os.getenv("DB_FILE", "nexus_signals.db")

# =========================
# Governance & AI Learning
# =========================
PERFORMANCE_FILE = "performance.json"
KILL_THRESHOLD = 45.0 
RECOVERY_THRESHOLD = 55.0

# =========================
# Webhook
# =========================
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
