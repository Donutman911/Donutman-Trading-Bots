# Donutman Trading Bots

Python backtesting and Alpaca paper-trading dashboard for the SMA and RSI strategies.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create `credentials/.env` from `credentials/.env.example`:

```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

The credentials file is ignored and must never be committed.

## Run the dashboard

```bash
.venv/bin/streamlit run strategy_dashboard.py
```

Open `http://localhost:8501`. The dashboard discovers compatible strategies in the project folder. A strategy must expose:

```python
fetch_data()
generate_signals()
backtest()
```

Included strategies:

- `Stategy_200-20_SMA.py`: 20/200 moving-average crossover
- `Strategy_BTC_USD.py`: BTC/USD SMA strategy
- `Strategy_RSI_30_70.py`: RSI strategy using 30/70 thresholds

The Alpaca controls are paper-only and support a small open/close test trade flow. Verify the selected endpoint is the paper endpoint before submitting orders.

## Run the 15-minute RSI paper bot

Run one evaluation manually:

```bash
.venv/bin/python rsi_paper_bot.py
```

The bot reads Alpaca 15-minute BTC/USD bars, calculates RSI 14, opens `0.001 BTC` when RSI is below 30, and closes the long position when RSI is above 70. It refuses to run against a non-paper endpoint.

To schedule it on macOS every 15 minutes:

```bash
cp com.donutman.rsi-paper-bot.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.donutman.rsi-paper-bot.plist
```

Check logs with:

```bash
tail -f rsi_paper_bot.log
```

Stop the schedule with:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.donutman.rsi-paper-bot.plist
```

## Run a strategy directly

```bash
.venv/bin/python Stategy_200-20_SMA.py
```