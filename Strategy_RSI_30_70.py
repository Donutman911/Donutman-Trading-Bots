import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from credentials.alpaca import get_alpaca_credentials

# ====================== CONFIG ======================
TICKER = "BTC/USD"
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70
COMMISSION = 0.001
INITIAL_CASH = 100_000
# ====================================================


def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def calculate_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def generate_signals(df, fast=RSI_PERIOD, slow=OVERBOUGHT, hold_bars=5, allow_short=True):
    df = df.copy()
    df["rsi"] = calculate_rsi(df["Close"], RSI_PERIOD)
    df["rsi_oversold"] = df["rsi"] < OVERSOLD
    df["rsi_overbought"] = df["rsi"] > OVERBOUGHT

    position = 0
    bars_held = 0
    positions = []

    for index in range(len(df)):
        if position != 0:
            bars_held += 1
            if bars_held >= hold_bars:
                position = 0
                bars_held = 0

        if df["rsi_oversold"].iloc[index] and position <= 0:
            position = 1
            bars_held = 0
        elif df["rsi_overbought"].iloc[index]:
            if allow_short and position >= 0:
                position = -1
                bars_held = 0
            elif not allow_short and position > 0:
                position = 0
                bars_held = 0

        positions.append(position)

    df["position"] = positions
    return df


def backtest(df, commission=COMMISSION, initial_cash=INITIAL_CASH):
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]
    position_change = df["position"].diff().abs()
    df["strategy_returns"] -= position_change * commission
    df["equity"] = initial_cash * (1 + df["strategy_returns"]).cumprod()
    return df


if __name__ == "__main__":
    data = fetch_data(TICKER, START_DATE, END_DATE)
    signals = generate_signals(data)
    results = backtest(signals)
    print(results[["Close", "rsi", "position", "equity"]].tail(10).round(2))
