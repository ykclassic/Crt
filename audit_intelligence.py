import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd

from config import (
    DB_FILE,
    PERFORMANCE_FILE,
    KILL_THRESHOLD,
    RECOVERY_THRESHOLD,
    EXCHANGE_ID,
)


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(
    BASE_DIR,
    DB_FILE,
)

PERFORMANCE_PATH = os.path.join(
    BASE_DIR,
    PERFORMANCE_FILE,
)

MAX_SIGNALS_PER_ENGINE = 30
OHLCV_LIMIT = 100

SIGNAL_TIME_TOLERANCE_MS = 1_000


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | AUDIT | %(levelname)s | %(message)s",
)


# ============================================================
# Exchange
# ============================================================

if EXCHANGE_ID not in ccxt.exchanges:
    raise EnvironmentError(
        f"Exchange '{EXCHANGE_ID}' is not supported by "
        f"the installed CCXT version."
    )

exchange_class = getattr(ccxt, EXCHANGE_ID)

exchange = exchange_class(
    {
        "enableRateLimit": True,
        "timeout": 15_000,
    }
)


# ============================================================
# Timestamp Helpers
# ============================================================

def parse_timestamp(timestamp: str) -> Optional[int]:
    """
    Convert an ISO-8601/SQLite timestamp to epoch milliseconds.
    """
    if not timestamp:
        return None

    try:
        normalized = timestamp.strip()

        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        dt = datetime.fromisoformat(normalized)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return int(dt.timestamp() * 1000)

    except ValueError:
        logging.warning(
            "Unable to parse timestamp: %s",
            timestamp,
        )
        return None


# ============================================================
# Market Outcome
# ============================================================

def get_market_outcome(
    ex,
    symbol: str,
    timeframe: str,
    start_ts: str,
    tp: float,
    sl: float,
    direction: str,
) -> str:
    """
    Determine the outcome of a signal using candles that occur
    after the signal timestamp.

    If TP and SL occur in the same candle, the result is marked
    PENDING because OHLCV data cannot establish intrabar order.
    """
    try:
        if tp is None or sl is None:
            return "PENDING"

        signal_ts_ms = parse_timestamp(start_ts)

        if signal_ts_ms is None:
            return "ERROR"

        ohlcv = ex.fetch_ohlcv(
            symbol,
            timeframe,
            since=signal_ts_ms,
            limit=OHLCV_LIMIT,
        )

        direction_upper = direction.upper()

        for candle in ohlcv:
            candle_timestamp = candle[0]
            high = float(candle[2])
            low = float(candle[3])

            if (
                candle_timestamp + SIGNAL_TIME_TOLERANCE_MS
                < signal_ts_ms
            ):
                continue

            if direction_upper == "LONG":

                hit_tp = high >= float(tp)
                hit_sl = low <= float(sl)

            elif direction_upper == "SHORT":

                hit_tp = low <= float(tp)
                hit_sl = high >= float(sl)

            else:
                logging.error(
                    "Unknown direction '%s' for %s",
                    direction,
                    symbol,
                )
                return "ERROR"

            # Ambiguous OHLC candle.
            if hit_tp and hit_sl:
                logging.warning(
                    "Ambiguous candle detected for %s %s "
                    "signal at %s: both TP and SL were reached.",
                    symbol,
                    timeframe,
                    start_ts,
                )
                return "PENDING"

            if hit_tp:
                return "WIN"

            if hit_sl:
                return "LOSS"

        return "PENDING"

    except Exception as exc:
        logging.error(
            "Market check error for %s %s: %s",
            symbol,
            timeframe,
            exc,
        )
        return "ERROR"


# ============================================================
# Performance File
# ============================================================

def load_existing_performance() -> dict:
    """
    Load the previous performance state.
    """
    if not os.path.exists(PERFORMANCE_PATH):
        return {}

    try:
        with open(
            PERFORMANCE_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        logging.warning(
            "Unable to load existing performance file: %s",
            exc,
        )

    return {}


# ============================================================
# Database
# ============================================================

def load_recent_signals() -> pd.DataFrame:
    """
    Load signals generated within the last seven days.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    query = """
        SELECT *
        FROM signals
        WHERE timestamp > datetime('now', '-7 days')
        ORDER BY timestamp ASC
    """

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )

    try:
        return pd.read_sql_query(
            query,
            conn,
        )

    finally:
        conn.close()


# ============================================================
# Required Schema
# ============================================================

def validate_signal_schema(
    df: pd.DataFrame,
) -> None:
    """
    Validate that all columns required by the audit engine
    exist before processing.
    """
    required_columns = {
        "engine_id",
        "symbol",
        "timeframe",
        "timestamp",
        "take_profit",
        "stop_loss",
        "direction",
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Signals database is missing required columns: "
            + ", ".join(missing)
        )


# ============================================================
# Engine Audit
# ============================================================

def audit_engine(
    engine_df: pd.DataFrame,
    engine_name: str,
) -> dict:
    """
    Audit the most recent signals for one strategy engine.
    """
    engine_df = (
        engine_df
        .sort_values("timestamp")
        .tail(MAX_SIGNALS_PER_ENGINE)
    )

    outcomes: list[str] = []

    for _, row in engine_df.iterrows():
        outcome = get_market_outcome(
            exchange,
            str(row["symbol"]),
            str(row["timeframe"]),
            str(row["timestamp"]),
            row["take_profit"],
            row["stop_loss"],
            str(row["direction"]),
        )

        outcomes.append(outcome)

    completed = [
        outcome
        for outcome in outcomes
        if outcome in {"WIN", "LOSS"}
    ]

    wins = completed.count("WIN")
    losses = completed.count("LOSS")

    total = len(completed)

    win_rate = (
        wins / total * 100
        if total > 0
        else 0.0
    )

    return {
        "win_rate": round(win_rate, 2),
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "pending": outcomes.count("PENDING"),
        "errors": outcomes.count("ERROR"),
        "status": None,
        "last_updated": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================
# Governance Status
# ============================================================

def determine_status(
    previous_status: str,
    win_rate: float,
    total_trades: int,
) -> str:
    """
    Apply the Nexus engine governance state machine.

    Status transitions:

        LIVE -> RECOVERY
        RECOVERY -> LIVE

    A minimum sample size is required before changing state.
    """
    status = previous_status

    if total_trades < 5:
        return status

    if win_rate < KILL_THRESHOLD:
        return "RECOVERY"

    if (
        previous_status == "RECOVERY"
        and win_rate >= RECOVERY_THRESHOLD
    ):
        return "LIVE"

    return status


# ============================================================
# Main Audit
# ============================================================

def run_audit() -> None:
    """
    Run the complete seven-day performance audit.
    """
    logging.info(
        "--- STARTING PERFORMANCE AUDIT ---"
    )

    try:
        df = load_recent_signals()

    except FileNotFoundError as exc:
        logging.error(str(exc))
        return

    except Exception as exc:
        logging.error(
            "Failed to load signals database: %s",
            exc,
        )
        return

    if df.empty:
        logging.info(
            "No signals found in the last 7 days to audit."
        )
        return

    try:
        validate_signal_schema(df)

    except RuntimeError as exc:
        logging.error(str(exc))
        return

    logging.info(
        "Loaded %d signals for audit.",
        len(df),
    )

    current_performance = (
        load_existing_performance()
    )

    performance = {}

    engines = (
        df["engine_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    logging.info(
        "Auditing %d engine(s).",
        len(engines),
    )

    for engine in engines:
        engine_df = df[
            df["engine_id"].astype(str) == engine
        ]

        result = audit_engine(
            engine_df,
            engine,
        )

        previous_status = (
            current_performance
            .get(engine, {})
            .get("status", "LIVE")
        )

        result["status"] = determine_status(
            previous_status=previous_status,
            win_rate=result["win_rate"],
            total_trades=result["total_trades"],
        )

        performance[engine] = result

        logging.info(
            "Engine=%s | win_rate=%.2f%% | "
            "trades=%d | wins=%d | losses=%d | "
            "pending=%d | status=%s",
            engine,
            result["win_rate"],
            result["total_trades"],
            result["wins"],
            result["losses"],
            result["pending"],
            result["status"],
        )

    try:
        with open(
            PERFORMANCE_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                performance,
                file,
                indent=4,
            )

    except OSError as exc:
        logging.error(
            "Failed to write performance file: %s",
            exc,
        )
        return

    logging.info(
        "Audit complete. Results saved to %s",
        PERFORMANCE_PATH,
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    run_audit()
