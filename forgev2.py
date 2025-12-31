# =========================
# MAIN ASYNC LOOP WITH SESSION & TIME
# =========================
async def main():
    exchange_cls = getattr(ccxtpro, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    await exchange.load_markets()
    df = pd.DataFrame()

    # Fetch initial OHLCV
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
        df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
    except: pass

    chart_placeholder = st.empty()
    signal_placeholder = st.empty()
    info_placeholder = st.empty()
    time_placeholder = st.empty()  # Placeholder for current time & session

    while True:
        # -------------------
        # CURRENT TIME & SESSION
        # -------------------
        now = datetime.now(timezone.utc)
        hour = now.hour
        if 0 <= hour < 8:
            session_name, note, color = "Asian Session", "Range-bound moves", "#FF8E53"
        elif 8 <= hour < 12:
            session_name, note, color = "London Open", "Breakouts expected", "#667eea"
        elif 12 <= hour < 16:
            session_name, note, color = "NY + London Overlap", "Highest volatility", "#764ba2"
        elif 16 <= hour < 21:
            session_name, note, color = "New York Session", "Trend continuation", "#43E97B"
        else:
            session_name, note, color = "Quiet Hours", "Low volume", "#888888"

        time_placeholder.markdown(
            f"<h4 style='text-align:center;color:{color};'>⏱ UTC Time: {now.strftime('%H:%M:%S')} — {session_name} ({note})</h4>",
            unsafe_allow_html=True
        )

        # -------------------
        # FETCH LIVE PRICE
        # -------------------
        try:
            ticker = await exchange.watch_ticker(symbol)
            price = ticker['last']
        except:
            price = None

        if price:
            new_row = {"Open": price, "High": price, "Low": price, "Close": price, "Volume": ticker.get("quoteVolume", 0)}
            df = df.append(pd.DataFrame([new_row], index=[datetime.utcnow()]))
            if len(df) > 300: df = df.iloc[-300:]

            signal, score, levels = generate_signal(df)

            # Signal display
            signal_placeholder.markdown(f"<h2 style='text-align:center;color:#00FF9F;'>Signal: {signal} ({score}%)</h2>", unsafe_allow_html=True)

            # Position info
            if levels:
                size, risk_amount = calc_position(levels, balance, risk_pct)
                info_placeholder.write(
                    f"Entry: ${levels['entry']:.2f}, SL: ${levels['sl']:.2f}, TP1: ${levels['tp1']:.2f}, TP2: ${levels['tp2']:.2f}, ATR: {levels['atr']:.2f}"
                )
                info_placeholder.write(f"Recommended Size: {size} {symbol.split('/')[0]}, Risk: ${risk_amount:.2f}")

            # Candlestick chart
            fig = go.Figure()
            fig.add_candlestick(
                x=df.index[-100:],
                open=df["Open"][-100:],
                high=df["High"][-100:],
                low=df["Low"][-100:],
                close=df["Close"][-100:],
                name="Price"
            )
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        await asyncio.sleep(1)
