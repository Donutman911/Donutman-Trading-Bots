"""Run one 15-minute BTC/USD RSI paper-trading evaluation.

Use macOS launchd to invoke this script every 900 seconds. The script is
intentionally one-shot so each run evaluates one completed 15-minute bar.
"""

from datetime import datetime, timedelta, timezone
import logging

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from credentials.alpaca import get_alpaca_credentials


SYMBOL = "BTC/USD"
RSI_PERIOD = 14
OVERSOLD = 30
OVERBOUGHT = 70
ORDER_QUANTITY = 0.001
BAR_TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def calculate_rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    average_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def load_recent_bars(data_client):
    end = datetime.now(timezone.utc) - timedelta(minutes=15)
    start = end - timedelta(days=5)
    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=BAR_TIMEFRAME,
        start=start,
        end=end,
        limit=200,
    )
    bars = data_client.get_crypto_bars(request).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL, level="symbol")
    return bars.sort_index()


def find_position(trading_client):
    normalized_symbol = SYMBOL.replace("/", "")
    for position in trading_client.get_all_positions():
        if position.symbol.replace("/", "") == normalized_symbol:
            return position
    return None


def submit_order(trading_client, side, quantity):
    order = MarketOrderRequest(
        symbol=SYMBOL,
        qty=quantity,
        side=side,
        time_in_force=TimeInForce.GTC,
    )
    return trading_client.submit_order(order_data=order)


def evaluate_once():
    api_key, secret_key, base_url = get_alpaca_credentials()
    if "paper-api.alpaca.markets" not in base_url:
        raise RuntimeError("Refusing to trade: ALPACA_BASE_URL is not the paper endpoint.")

    data_client = CryptoHistoricalDataClient(api_key, secret_key)
    trading_client = TradingClient(api_key, secret_key, paper=True)
    bars = load_recent_bars(data_client)
    if len(bars) < RSI_PERIOD + 2:
        raise RuntimeError("Not enough 15-minute bars to calculate RSI.")

    bars["rsi"] = calculate_rsi(bars["close"])
    latest = bars.iloc[-1]
    rsi = float(latest["rsi"])
    position = find_position(trading_client)

    logging.info("%s 15m close=%.2f RSI=%.2f position=%s", SYMBOL, latest["close"], rsi, bool(position))

    if rsi < OVERSOLD and position is None:
        order = submit_order(trading_client, OrderSide.BUY, ORDER_QUANTITY)
        logging.info("OPENED paper long: order=%s qty=%s", order.id, ORDER_QUANTITY)
    elif rsi > OVERBOUGHT and position is not None:
        quantity = abs(float(position.qty))
        order = submit_order(trading_client, OrderSide.SELL, quantity)
        logging.info("CLOSED paper long: order=%s qty=%s", order.id, quantity)
    else:
        logging.info("No trade: RSI remains between the action thresholds or position state is unchanged.")


if __name__ == "__main__":
    evaluate_once()
