# =========================
# MAIN ASYNC LOOP WITH DYNAMIC SESSION DISPLAY
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
    session_placeholder = st.empty()  # Placeholder for session & UTC time

    while True:
        # -------------------
        # CURRENT TIME & SESSION
        # -------------------
        now = datetime.now(timezone.utc)
        hour = now.hour

        if 0 <= hour < 8:
            session_name, note, color, pulse = "Asian Session", "Range-bound moves", "#FFB86C", "0.8"
        elif 8 <= hour < 12:
            session_name, note, color, pulse = "London Open", "Breakouts expected", "#8A2BE2", "1"
        elif 12 <= hour < 16:
            session_name, note, color, pulse = "NY + London Overlap", "Highest volatility", "#FF416C", "1.2"
        elif 16 <= hour < 21:
            session_name, note, color, pulse = "New York Session", "Trend continuation", "#43E97B", "1"
        else:
            session_name, note, color, pulse = "Quiet Hours", "Low volume", "#888888", "0.6"

        session_placeholder.markdown(f"""
        <div style="
            text-align:center;
            padding:10px 0;
            border-radius:15px;
            background: linear-gradient(90deg, {color} 0%, #000000 100%);
            color:white;
            font-size:1.5rem;
            font-weight:bold;
            animation: pulse {pulse}s infinite alternate;
        ">
            ⏱ UTC Time: {now.strftime('%H:%M:%S')} — {session_name} ({note})
        </div>
        <style>
        @keyframes pulse {{
            0% {{ opacity: 0.6; }}
            100% {{ opacity: 1; }}
        }}
        </style>
        """, unsafe_allow_html=True)

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
