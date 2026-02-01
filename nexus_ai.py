import ccxt
import pandas as pd

def get_market_trend(node, symbol):
    """Prevents LONG signals in a bearish market."""
    bars = node.fetch_ohlcv(symbol, timeframe='1h', limit=50)
    df = pd.DataFrame(bars, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    sma = df['c'].rolling(window=20).mean().iloc[-1]
    current_price = df['c'].iloc[-1]
    # Return 'BEAR' if price is below average, 'BULL' if above
    return 'BULL' if current_price > sma else 'BEAR'

# In your signal generation loop:
# trend = get_market_trend(node, asset)
# if trend == 'BEAR':
#    signal = 'SHORT'  # Force pivot to Short or Skip
