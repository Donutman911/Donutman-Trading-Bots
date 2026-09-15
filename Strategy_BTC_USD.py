import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from credentials.alpaca import get_alpaca_credentials

# ====================== CONFIG ======================
TICKER          = "BTC/USD"          # Change to any ticker
START_DATE      = "2015-01-01"
END_DATE        = datetime.today().strftime("%Y-%m-%d")
FAST_MA        = 20
SLOW_MA        = 200
HOLD_BARS      = 5              # Maximum bars to hold after entry
ALLOW_SHORT    = True           # Set False for long-only
COMMISSION      = 0.001        # 0.1% per side
INITIAL_CASH    = 100_000
# ====================================================

def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def generate_signals(df, fast=20, slow=200, hold_bars=5, allow_short=True):
    df = df.copy()
    df['sma_fast'] = df['Close'].rolling(fast).mean()
    df['sma_slow'] = df['Close'].rolling(slow).mean()

    # Crossover detection (no look-ahead)
    df['cross_up']  = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))
    df['cross_down'] = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))

    # Position logic
    position = 0          # 1 = long, -1 = short, 0 = flat
    bars_held = 0
    positions = []

    for i in range(len(df)):
        # Force exit after hold_bars
        if position != 0:
            bars_held += 1
            if bars_held >= hold_bars:
                position = 0
                bars_held = 0

        # New signals (only act if currently flat or opposite signal)
        if df['cross_up'].iloc[i]:
            if position <= 0:          # enter/flip to long
                position = 1
                bars_held = 0
        elif df['cross_down'].iloc[i]:
            if allow_short and position >= 0:
                position = -1
                bars_held = 0
            elif not allow_short and position > 0:
                position = 0
                bars_held = 0

        positions.append(position)

    df['position'] = positions
    return df

def backtest(df, commission=0.001, initial_cash=100_000):
    df = df.copy()
    df['returns'] = df['Close'].pct_change()

    # Strategy returns (shift position so we trade on next bar open conceptually)
    df['strategy_returns'] = df['position'].shift(1) * df['returns']

    # Apply commission on position changes
    position_change = df['position'].diff().abs()
    df['strategy_returns'] -= position_change * commission

    # Equity curve
    df['equity'] = initial_cash * (1 + df['strategy_returns']).cumprod()

    # Performance metrics
    total_return = df['equity'].iloc[-1] / initial_cash - 1
    buy_hold = df['Close'].iloc[-1] / df['Close'].iloc[0] - 1

    # Annualized numbers (assuming ~252 trading days)
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    cagr = (df['equity'].iloc[-1] / initial_cash) ** (1 / years) - 1 if years > 0 else 0

    sharpe = np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std() \
            if df['strategy_returns'].std() != 0 else 0

    max_dd = (df['equity'] / df['equity'].cummax() - 1).min()

    print("=" * 50)
    print(f"Backtest results for {TICKER}")
    print("=" * 50)
    print(f"Period          : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Total Return    : {total_return: .2%}")
    print(f"Buy & Hold      : {buy_hold: .2%}")
    print(f"CAGR            : {cagr: .2%}")
    print(f"Sharpe Ratio    : {sharpe: .2f}")
    print(f"Max Drawdown    : {max_dd: .2%}")
    print(f"Final Equity    : ${df['equity'].iloc[-1]:,.0f}")
    print("=" * 50)

    return df

# ====================== RUN ======================
if __name__ == "__main__":
    data = fetch_data(TICKER, START_DATE, END_DATE)
    signals = generate_signals(data, FAST_MA, SLOW_MA, HOLD_BARS, ALLOW_SHORT)
    results = backtest(signals, COMMISSION, INITIAL_CASH)

    # Optional: show last few signals
    print("\nLast 10 rows with signals:")
    print(results[['Close', 'sma_fast', 'sma_slow', 'position']].tail(10).round(2))
