import os

# =========================
# Exchange
# =========================
EXCHANGE_ID = "gateio"

# =========================
# Assets (Override via GitHub Secret if needed)
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
DB_FILE = os.getenv("DB_FILE", "signals.db")

# =========================
# Governance & AI Learning
# =========================
# The file where the AI stores engine performance stats
PERFORMANCE_FILE = "performance.json"

# Win rate % below which an engine is put in 'RECOVERY' (Learning mode)
KILL_THRESHOLD = 45.0 

# Win rate % required to restore an engine to 'LIVE' status
RECOVERY_THRESHOLD = 55.0

# =========================
# Webhook
# =========================
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
