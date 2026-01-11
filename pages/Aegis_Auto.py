# ==========================
# MULTI-ASSET SIGNAL SCAN (ROBUST + REASON LOGGING)
# ==========================

# 20 trading pairs
ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT",
    "ADA/USDT","LINK/USDT","TRX/USDT","SUI/USDT","PEPE/USDT",
    "BNB/USDT","MATIC/USDT","LTC/USDT","AVAX/USDT","UNI/USDT",
    "ATOM/USDT","NEAR/USDT","FTM/USDT","ALGO/USDT","VET/USDT"
]

signals = []

# Safe fetch wrapper
def safe_fetch(symbol, timeframe, limit=400):
    try:
        df = fetch_ohlcv(symbol, timeframe, limit)
        if df.empty or set(["ts","o","h","l","c","v","dt"]) - set(df.columns):
            st.warning(f"{symbol} {timeframe} OHLCV missing or incomplete")
            return pd.DataFrame(columns=["ts","o","h","l","c","v","dt"])
        return df
    except Exception as e:
        st.warning(f"{symbol} {timeframe} fetch failed: {e}")
        return pd.DataFrame(columns=["ts","o","h","l","c","v","dt"])

for asset in ASSETS:
    reason = ""

    # Fetch features safely
    df_1h = features(safe_fetch(asset, "1h"))
    df_4h = features(safe_fetch(asset, "4h"))

    # Skip if empty
    if df_1h.empty or df_4h.empty:
        st.info(f"{asset}: OHLCV data missing for 1H or 4H")
        continue

    # Ensemble predictions and confidence
    pred_1h, conf_1h = ensemble_signal(df_1h)
    pred_4h, conf_4h = ensemble_signal(df_4h)

    # Determine direction
    dir_1h = "LONG" if pred_1h > df_1h["c"].iloc[-1] else "SHORT"
    dir_4h = "LONG" if pred_4h > df_4h["c"].iloc[-1] else "SHORT"

    # Track reason for filtering
    if conf_1h <= 60:
        reason = f"1H confidence too low ({conf_1h:.2f}%)"
    elif conf_4h <= 60:
        reason = f"4H confidence too low ({conf_4h:.2f}%)"
    elif dir_1h != dir_4h:
        reason = f"Direction mismatch (1H={dir_1h}, 4H={dir_4h})"

    # Generate signal if all filters pass
    if not reason:
        signal = build_signal(
            df_1h,
            pred_1h,
            (conf_1h + conf_4h)/2,
            asset,
            "1H (4H Confirmed)"
        )
        if signal:
            signals.append(signal)
            # Optional: Telegram alert
            send_telegram(
                f"""AEGIS SIGNAL (ANALYTICAL)
Asset: {signal['asset']}
Timeframe: {signal['timeframe']}
Bias: {signal['bias']}
Reference Price: {signal['price']:.2f}
Projected Objective: {signal['objective']:.2f}
Invalidation Level: {signal['invalidation']:.2f}
Confidence: {signal['confidence']:.2f}%
Regime: {signal['regime']}"""
            )

    # Log reason if signal was not generated
    if reason:
        st.info(f"{asset}: No signal generated → {reason}")
        # Optional: store in DB
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals_log (
            timestamp TEXT,
            asset TEXT,
            reason TEXT
        )
        """)
        cursor.execute("""
        INSERT INTO signals_log (timestamp, asset, reason)
        VALUES (?,?,?)
        """, (datetime.utcnow().isoformat(), asset, reason))
        conn.commit()

# Display generated signals
if signals:
    st.subheader("📊 Qualified Signals")
    st.dataframe(pd.DataFrame(signals))
else:
    st.warning("No qualified signals under current market conditions. Check reasons above.")
