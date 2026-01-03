import ccxt
import pandas as pd
import numpy as np

def scan_volatility_volume(pairs, exchange_id='bitget', timeframe='1d', days=30):
    """
    Scans the given pairs for volatility and average volume.
    
    - Volatility: Annualized % from daily log returns.
    - Avg Volume: Mean daily volume over the period.
    """
    exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True})
    results = []

    for pair in pairs:
        try:
            ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=days + 10)  # Extra for safety
            if len(ohlcv) < 20:
                results.append({ 'Pair': pair, 'Volatility (%)': 'N/A (insufficient data)', 'Avg Daily Volume': 'N/A' })
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            volatility = df['log_return'].std() * np.sqrt(365) * 100  # Annualized volatility %
            avg_volume = df['volume'].mean()

            results.append({
                'Pair': pair,
                'Volatility (%)': round(volatility, 2),
                'Avg Daily Volume': f"{avg_volume:,.0f}"
            })
        except Exception as e:
            results.append({ 'Pair': pair, 'Volatility (%)': 'Error', 'Avg Daily Volume': str(e) })

    # Sort by volatility descending
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Volatility (%)', ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
    
    print(results_df.to_string(index=False))
    return results_df

# === YOUR SELECTED PAIRS ===
selected_pairs = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "LTC/USDT", "DOGE/USDT", "SHIB/USDT", "PEPE/USDT",
    "TRX/USDT", "LINK/USDT", "TON/USDT", "AVAX/USDT", "DOT/USDT",
    "MATIC/USDT", "UNI/USDT", "AAVE/USDT", "NEAR/USDT", "SUI/USDT"
]

if __name__ == "__main__":
    print("Scanning on Bitget (as of January 3, 2026)...\n")
    scan_volatility_volume(selected_pairs)
