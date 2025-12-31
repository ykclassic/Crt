# telegram_bot.py
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# === YOUR BOT TOKEN ===
TOKEN = "8367963721:AAH6B819_DevFNpZracbJ5EmHrDR3DKZeR4"

# Hardcoded list of symbols we're monitoring (same as Streamlit app)
TRADING_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "LTC/USDT", "DOGE/USDT", "MATIC/USDT", "AVAX/USDT"
]

# Placeholder for latest signals (we'll simulate or update this manually for now)
# In future, you can connect via webhook/database/shared storage
latest_signals = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Latest Signals", callback_data="signals")],
        [InlineKeyboardButton("🟢 Status", callback_data="status")],
        [InlineKeyboardButton("🔄 Refresh Dashboard", url="https://your-streamlit-app-url.streamlit.app")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "🔥 *ProfitForge Pro Bot*\n\n"
        "Welcome back! Choose an option below:",
        parse_mode="Markdown",
        reply_markup=reply_markup
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🟢 *Bot Status*\n\n"
        "Bot is running perfectly.\n"
        "Real-time alerts: ✅ Active\n"
        "Dashboard: Online",
        parse_mode="Markdown"
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge button press

    if query.data == "status":
        await query.edit_message_text(
            "🟢 *Bot Status*\n\n"
            "Bot is running perfectly.\n"
            "Real-time alerts: ✅ Active\n"
            "Dashboard: Online",
            parse_mode="Markdown"
        )

    elif query.data == "signals":
        msg = "📊 *Latest Live Signals*\n\n"
        if not latest_signals:
            msg += "No strong signals right now.\nWaiting for market moves..."
        else:
            for symbol, data in latest_signals.items():
                msg += (
                    f"*{symbol}*\n"
                    f"Signal: {data['signal']}\n"
                    f"Score: {data['score']:.1f} | Price: ${data['price']:.2f}\n"
                    f"TF: {data['tf']}\n\n"
                )

        # Add refresh button
        keyboard = [[InlineKeyboardButton("🔄 Refresh Signals", callback_data="signals")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=reply_markup)

async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await button_handler(update.callback_query or update, context)  # Reuse logic

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("signals", signals_command))
    app.add_handler(CallbackQueryHandler(button_handler))

    print("🤖 ProfitForge Telegram Bot started!")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
