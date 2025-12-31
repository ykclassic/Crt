import streamlit as st
import asyncio
import ccxtpro
import pandas as pd
from datetime import datetime, timezone
import plotly.graph_objects as go

st.set_page_config(page_title="ProfitForge Pro", layout="wide")

# Placeholders
session_ph = st.empty()
price_ph = st.empty()
signal_ph = st.empty()
chart_ph = st.empty()

# --- Async live loop ---
async def main_loop():
    exchange_id = "binance"
    symbol = "BTC/USDT"
    timeframe = "1h"
    balance = 10000
    risk_pct = 1.0

    exchange_cls = getattr(ccxtpro, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    await exchange.load_markets()
    df = pd.DataFrame()

    while True:
        # --- Live UTC & Session ---
        now = datetime.now(timezone.utc)
        hour = now.hour
        if 0 <= hour < 8:
            session_name, color = "Asian Session", "#FFB86C"
        elif 8 <= hour < 12:
            session_name, color = "London Open", "#8A2BE2"
        elif 12 <= hour < 16:
            session_name, color = "NY + London Overlap", "#FF416C"
        elif 16 <= hour < 21:
            session_name, color = "New York Session", "#43E97B"
        else:
            session_name, color = "Quiet Hours", "#888888"

        session_ph.markdown(
            f"<div style='text-align:center;color:white;background:{color};padding:8px;border-radius:12px;'>"
            f"UTC {now.strftime('%H:%M:%S')} — {session_name}</div>",
            unsafe_allow_html=True
        )

        # --- Fetch live price ---
        try:
            ticker = await exchange.watch_ticker(symbol)
            price = ticker["last"]
        except:
            price = None

        if price:
            price_ph.markdown(f"<h2 style='text-align:center;color:#00FF9F;'>Live Price: ${price:,.2f}</h2>", unsafe_allow_html=True)

            # Append price to df
            new_row = {"Open": price, "High": price, "Low": price, "Close": price, "Volume": ticker.get("quoteVolume", 0)}
            df = df.append(pd.DataFrame([new_row], index=[datetime.utcnow()]))
            if len(df) > 300: df = df.iloc[-300:]

            # --- Generate signal here ---
            # For demo, use a dummy signal
            signal = "BUY" if price%2==0 else "HOLD"
            signal_ph.markdown(f"<h2 style='text-align:center;color:#00FF9F;'>Signal: {signal}</h2>", unsafe_allow_html=True)

            # --- Candlestick chart ---
            fig = go.Figure()
            fig.add_candlestick(
                x=df.index[-50:],
                open=df["Open"][-50:], high=df["High"][-50:],
                low=df["Low"][-50:], close=df["Close"][-50:]
            )
            fig.update_layout(height=400, xaxis_rangeslider_visible=False)
            chart_ph.plotly_chart(fig, use_container_width=True)

        await asyncio.sleep(1)  # Update every second

# --- Run async loop ---
if __name__ == "__main__":
    asyncio.run(main_loop())
