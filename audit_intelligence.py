import sqlite3
import pandas as pd
import ccxt
import json
import os
from datetime import datetime

# --- CONFIGURATION ---
DB_FILES = {
    "nexus_core": "nexus_core.db",
    "hybrid_v1": "hybrid_v1.db",
    "rangemaster": "rangemaster.db",
    "nexus_ai": "nexus_ai.db"
}
PERFORMANCE_FILE = "performance.json"

# Thresholds for the Kill Switch
KILL_THRESHOLD = 40.0    # Drop below this → RECOVERY (on closed trades)
RECOVERY_THRESHOLD = 50.0 # Climb above this → LIVE

# Audit settings
MAX_SIGNALS_PER_ENGINE = 100
MAX_CANDLES_LOOKFORWARD = 500  # ~3 weeks on 1h, prevents hanging on very wide SL/TP

def determine_outcome(row, ex):
    """Fetch historical candles after signal and check TP/SL hit order."""
    try:
        signal_ts = datetime.fromisoformat(row['ts'])
        since_ms = int(signal_ts.timestamp() * 1000) + 1  # Start from next millisecond

        # Fetch candles after signal
        ohlcv = ex.fetch_ohlcv(
            row['asset'],
            timeframe=row['timeframe'],
            since=since_ms,
            limit=MAX_CANDLES_LOOKFORWARD
        )
        if not ohlcv:
            return "ongoing"

        df_candles = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df_candles['ts'] = pd.to_datetime(df_candles['ts'], unit='ms')

        entry = row['entry']
        sl = row['sl']
        tp = row['tp']
        direction = row['signal'].upper()

        for _, candle in df_candles.iterrows():
            if direction == "LONG":
                if candle['low'] <= sl:
                    return "loss"
                if candle['high'] >= tp:
                    return "win"
            else:  # SHORT
                if candle['high'] >= sl:
                    return "loss"
                if candle['low'] <= tp:
                    return "win"

        return "ongoing"  # Neither hit within lookforward
    except Exception as e:
        print(f"   ⚠️ Outcome check failed for {row['asset']} {row['ts']}: {e}")
        return "skipped"

def audit_all():
    ex = ccxt.xt({"enableRateLimit": True})
    performance_report = {}

    # Load existing performance
    if os.path.exists(PERFORMANCE_FILE):
        try:
            with open(PERFORMANCE_FILE, "r") as f:
                performance_report = json.load(f)
        except:
            performance_report = {}

    print(f"🔍 Starting intelligent audit across {len(DB_FILES)} engines...")

    for strat_id, db_path in DB_FILES.items():
        if not os.path.exists(db_path):
            print(f"   ⚠️ DB missing: {db_path}")
            continue

        conn = sqlite3.connect(db_path)
        try:
            # Load recent signals
            df = pd.read_sql_query(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
                conn, params=(MAX_SIGNALS_PER_ENGINE,)
            )
            if df.empty:
                print(f"   ℹ️ No signals in {strat_id}")
                continue

            # Ensure required columns exist and are numeric
            required = ['entry', 'sl', 'tp', 'confidence']
            if not all(col in df.columns for col in required):
                print(f"   ⚠️ Incomplete schema in {strat_id} - skipping audit")
                continue

            df['entry'] = pd.to_numeric(df['entry'], errors='coerce')
            df['sl'] = pd.to_numeric(df['sl'], errors='coerce')
            df['tp'] = pd.to_numeric(df['tp'], errors='coerce')
            df = df.dropna(subset=['entry', 'sl', 'tp', 'timeframe', 'signal'])

            wins = 0
            losses = 0
            ongoing = 0
            skipped = 0

            print(f"   Auditing {len(df)} recent signals for {strat_id}...")

            for _, row in df.iterrows():
                outcome = determine_outcome(row, ex)
                if outcome == "win":
                    wins += 1
                elif outcome == "loss":
                    losses += 1
                elif outcome == "ongoing":
                    ongoing += 1
                else:
                    skipped += 1

            closed_trades = wins + losses
            wr = round((wins / closed_trades * 100), 2) if closed_trades > 0 else 50.0
            sample_size = closed_trades

            # Status logic (kill switch)
            current_status = performance_report.get(strat_id, {}).get("status", "LIVE")
            if current_status == "LIVE" and wr < KILL_THRESHOLD and sample_size >= 5:
                new_status = "RECOVERY"
            elif current_status == "RECOVERY" and wr >= RECOVERY_THRESHOLD and sample_size >= 5:
                new_status = "LIVE"
            else:
                new_status = current_status

            performance_report[strat_id] = {
                "win_rate": wr,
                "status": new_status,
                "closed_sample": sample_size,
                "wins": wins,
                "losses": losses,
                "ongoing": ongoing,
                "last_audit": datetime.now().isoformat()
            }

            print(f"   ✅ {strat_id}: WR={wr}% ({wins}W/{losses}L, {ongoing} ongoing) → {new_status}")

        finally:
            conn.close()

    # Save updated performance
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(performance_report, f, indent=4)

    print(f"✅ Audit Complete. {len(performance_report)} engines updated.")

if __name__ == "__main__":
    audit_all()
