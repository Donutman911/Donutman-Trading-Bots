from datetime import date, timedelta
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from credentials.alpaca import get_alpaca_credentials


ROOT = Path(__file__).resolve().parent
REQUIRED_STRATEGY_FUNCTIONS = ("fetch_data", "generate_signals", "backtest")


def discover_strategies():
    return sorted(
        path
        for path in ROOT.glob("*.py")
        if path.name != Path(__file__).name and not path.name.startswith("__")
    )


@st.cache_resource
def load_strategy(strategy_path):
    strategy_file = Path(strategy_path)
    module_name = f"strategy_{strategy_file.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    missing_functions = [
        name for name in REQUIRED_STRATEGY_FUNCTIONS if not hasattr(strategy, name)
    ]
    if missing_functions:
        raise AttributeError(
            f"{strategy_file.name} is missing: {', '.join(missing_functions)}"
        )
    return strategy


@st.cache_data(show_spinner=False)
def load_market_data(strategy_path, ticker, start_date, end_date):
    return load_strategy(strategy_path).fetch_data(ticker, start_date, end_date)


@st.cache_resource(show_spinner=False)
def get_paper_trading_client():
    from alpaca.trading.client import TradingClient

    api_key, secret_key, base_url = get_alpaca_credentials()
    if "paper-api.alpaca.markets" not in base_url:
        raise RuntimeError(
            "Paper trading is disabled because ALPACA_BASE_URL is not the Alpaca paper endpoint."
        )
    return TradingClient(api_key, secret_key, paper=True)


def submit_paper_market_order(client, symbol, side, quantity):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    time_in_force = TimeInForce.GTC if "/" in symbol else TimeInForce.DAY
    order = MarketOrderRequest(
        symbol=symbol.upper(),
        qty=quantity,
        side=OrderSide.BUY if side == "Buy" else OrderSide.SELL,
        time_in_force=time_in_force,
    )
    return client.submit_order(order_data=order)


def submit_test_trade(client, symbol, quantity, opening):
    return submit_paper_market_order(
        client,
        symbol,
        "Buy" if opening else "Sell",
        quantity,
    )


def build_trades(results, commission):
    trades = []
    active = None
    previous_position = 0

    for timestamp, row in results.iterrows():
        position = int(row["position"])
        if position != previous_position:
            if active is not None:
                exit_price = float(row["Close"])
                gross_return = active["side"] * (exit_price / active["entry_price"] - 1)
                net_return = gross_return - commission * (1 + abs(active["side"]))
                trades.append(
                    {
                        "Entry": active["entry_date"],
                        "Exit": timestamp,
                        "Side": "Long" if active["side"] == 1 else "Short",
                        "Entry Price": active["entry_price"],
                        "Exit Price": exit_price,
                        "Return": net_return,
                        "Bars": (timestamp - active["entry_date"]).days,
                    }
                )
                active = None
            if position != 0:
                active = {
                    "entry_date": timestamp,
                    "entry_price": float(row["Close"]),
                    "side": position,
                }
        previous_position = position

    if active is not None:
        timestamp = results.index[-1]
        exit_price = float(results["Close"].iloc[-1])
        gross_return = active["side"] * (exit_price / active["entry_price"] - 1)
        trades.append(
            {
                "Entry": active["entry_date"],
                "Exit": timestamp,
                "Side": "Long" if active["side"] == 1 else "Short",
                "Entry Price": active["entry_price"],
                "Exit Price": exit_price,
                "Return": gross_return - commission * (1 + abs(active["side"])),
                "Bars": (timestamp - active["entry_date"]).days,
            }
        )

    return pd.DataFrame(trades)


def calculate_metrics(results, trades, initial_cash):
    strategy_returns = results["strategy_returns"].dropna()
    equity = results["equity"].dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total_return = equity.iloc[-1] / initial_cash - 1
    cagr = (equity.iloc[-1] / initial_cash) ** (1 / years) - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() else 0
    downside = strategy_returns[strategy_returns < 0].std() * np.sqrt(252)
    sortino = np.sqrt(252) * strategy_returns.mean() / strategy_returns[strategy_returns < 0].std() if strategy_returns[strategy_returns < 0].std() else 0
    drawdown = equity / equity.cummax() - 1
    wins = trades[trades["Return"] > 0] if not trades.empty else trades
    losses = trades[trades["Return"] <= 0] if not trades.empty else trades
    profit_factor = wins["Return"].sum() / abs(losses["Return"].sum()) if not losses.empty and losses["Return"].sum() else 0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": drawdown.min(),
        "Final Equity": equity.iloc[-1],
        "Buy & Hold": results["Close"].iloc[-1] / results["Close"].iloc[0] - 1,
        "Trades": len(trades),
        "Win Rate": len(wins) / len(trades) if len(trades) else 0,
        "Profit Factor": profit_factor,
        "Avg Trade": trades["Return"].mean() if not trades.empty else 0,
    }


def metric_card(label, value, number_format="percent"):
    if number_format == "percent":
        display_value = f"{value:.2%}"
    elif number_format == "money":
        display_value = f"${value:,.0f}"
    elif number_format == "decimal":
        display_value = f"{value:.2f}"
    else:
        display_value = f"{value:,.0f}"
    st.metric(label, display_value)


st.set_page_config(page_title="Strategy Lab", page_icon=":material/monitoring:", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink: #eef4f7; --muted: #8ea0b0; --line: #263546; --panel: #111923; --panel-bright: #172332; --cyan: #62d8ff; --violet: #b692ff; --amber: #ffbd69; --coral: #ff718d; --green: #58d6a3; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 86% 0%, rgba(75, 92, 184, .16), transparent 31rem), #080c12; }
    [data-testid="stHeader"] { background: rgba(8, 12, 18, .82); }
    [data-testid="stSidebar"] { background: #0e151e; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1 { font-size: 1.2rem; font-weight: 700; }
    [data-testid="stMetric"] { background: linear-gradient(135deg, rgba(23,35,50,.96), rgba(15,24,34,.96)); border: 1px solid var(--line); border-top: 2px solid rgba(98,216,255,.62); padding: 17px 18px; border-radius: 8px; box-shadow: 0 12px 28px rgba(0,0,0,.15); }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .78rem; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'DM Mono', monospace; font-size: 1.45rem; }
    [data-testid="stSidebar"] [data-testid="stButton"] button { border: 1px solid rgba(98,216,255,.55); box-shadow: 0 0 22px rgba(98,216,255,.12); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    [data-testid="stPlotlyChart"] { border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px 0; background: rgba(17,25,35,.62); }
    h1 { font-size: clamp(2rem, 4vw, 3.4rem); font-weight: 800; letter-spacing: -.04em; line-height: 1.05; }
    h2, h3 { letter-spacing: -.02em; }
    h2 { margin-top: 2rem; }
    .eyebrow { color: var(--cyan); font: 500 .72rem 'DM Mono', monospace; letter-spacing: .14em; text-transform: uppercase; }
    .caption { color: var(--muted); font-size: .92rem; }
    .section-kicker { color: var(--amber); font: 500 .68rem 'DM Mono', monospace; letter-spacing: .13em; text-transform: uppercase; margin-bottom: -.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='eyebrow'>Strategy Lab / Backtest</div>", unsafe_allow_html=True)
strategy_files = discover_strategies()
if not strategy_files:
    st.error("No strategy files were found in the workspace.")
    st.stop()

strategy_labels = {path: path.stem.replace("_", " ").replace("-", " ") for path in strategy_files}
selected_strategy_path = st.sidebar.selectbox(
    "Strategy",
    options=strategy_files,
    format_func=lambda path: strategy_labels[path],
    help="Strategies must expose fetch_data, generate_signals, and backtest functions.",
)
st.sidebar.title("Run parameters")
ticker = st.sidebar.text_input("Ticker", value="SPY").strip().upper()
start_date = st.sidebar.date_input("Start date", value=date(2015, 1, 1))
end_date = st.sidebar.date_input("End date", value=date.today())
fast_ma = st.sidebar.number_input("Fast moving average", min_value=2, max_value=500, value=20, step=1)
slow_ma = st.sidebar.number_input("Slow moving average", min_value=3, max_value=1000, value=200, step=1)
hold_bars = st.sidebar.number_input("Maximum holding bars", min_value=1, max_value=1000, value=5, step=1)
commission = st.sidebar.number_input("Commission per side", min_value=0.0, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
initial_cash = st.sidebar.number_input("Starting capital", min_value=1000.0, value=100000.0, step=1000.0)
allow_short = st.sidebar.toggle("Allow short positions", value=True)
run_backtest = st.sidebar.button("Run backtest", type="primary", width="stretch", icon=":material/play_arrow:")

if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if run_backtest:
    st.session_state.run_id += 1

st.title("Strategy Lab", icon=":material/monitoring:")
st.markdown("<p class='caption'>Compare systematic signals with transparent assumptions, paper execution, and repeatable runs.</p>", unsafe_allow_html=True)

if start_date >= end_date:
    st.error("The start date must be earlier than the end date.")
    st.stop()
if fast_ma >= slow_ma:
    st.warning("The fast moving average is usually smaller than the slow moving average.")

with st.spinner(f"Downloading {ticker} market data and running the backtest..."):
    try:
        strategy = load_strategy(str(selected_strategy_path))
        data = load_market_data(selected_strategy_path.as_posix(), ticker, start_date.isoformat(), (end_date + timedelta(days=1)).isoformat())
        if data.empty:
            st.error("No market data was returned for that ticker and date range.")
            st.stop()
        signals = strategy.generate_signals(data, int(fast_ma), int(slow_ma), int(hold_bars), allow_short)
        results = strategy.backtest(signals, commission, initial_cash)
        trades = build_trades(results, commission)
        metrics = calculate_metrics(results, trades, initial_cash)
    except Exception as error:
        st.error(f"Backtest failed: {error}")
        st.stop()

st.caption(f"{strategy_labels[selected_strategy_path]}  ·  {ticker}  ·  {results.index[0].date()} to {results.index[-1].date()}  ·  {len(results):,} trading sessions")

st.markdown("<div class='section-kicker'>live sandbox / alpaca paper account</div>", unsafe_allow_html=True)
with st.container(border=True):
    st.subheader("Paper trading", icon=":material/account_balance:")
    st.caption("Orders here use Alpaca's paper endpoint only. They do not use real money.")
    test_trade_message = st.session_state.pop("test_trade_message", None)
    if test_trade_message:
        st.success(test_trade_message)
    paper_client = None
    try:
        paper_client = get_paper_trading_client()
        account = paper_client.get_account()
        account_columns = st.columns(4)
        account_columns[0].metric("Account status", str(account.status).replace(".", " ").title())
        account_columns[1].metric("Paper equity", f"${float(account.equity):,.2f}")
        account_columns[2].metric("Buying power", f"${float(account.buying_power):,.2f}")
        account_columns[3].metric("Cash", f"${float(account.cash):,.2f}")

        positions = paper_client.get_all_positions()
        if positions:
            positions_table = pd.DataFrame(
                [
                    {
                        "Symbol": position.symbol,
                        "Side": str(position.side).replace(".", " ").title(),
                        "Quantity": float(position.qty),
                        "Market Value": float(position.market_value),
                        "Unrealized P/L": float(position.unrealized_pl),
                    }
                    for position in positions
                ]
            )
            st.dataframe(positions_table, width="stretch", hide_index=True)
        else:
            st.caption("No open paper positions.")

        st.markdown("**RSI test trade**")
        st.caption("One click opens a small paper position. The next click closes the position opened by this session.")
        st.session_state.setdefault("test_trade_symbol", ticker)
        st.session_state.setdefault("test_trade_quantity", 0.001 if "/" in ticker else 1.0)
        test_symbol = st.text_input("Test symbol", value=ticker, max_chars=10, key="test_trade_symbol_input").strip().upper()
        test_quantity = st.number_input(
            "Test quantity",
            min_value=0.000001 if "/" in test_symbol else 1.0,
            value=0.001 if "/" in test_symbol else 1.0,
            step=0.001 if "/" in test_symbol else 1.0,
            format="%.6f" if "/" in test_symbol else "%.0f",
            key="test_trade_quantity_input",
        )
        if "test_trade_active" not in st.session_state:
            normalized_test_symbol = test_symbol.replace("/", "")
            existing_test_position = next(
                (
                    position
                    for position in positions
                    if position.symbol.replace("/", "") == normalized_test_symbol
                    and str(position.side).lower().endswith("long")
                ),
                None,
            )
            st.session_state.test_trade_active = existing_test_position is not None
            if existing_test_position is not None:
                st.session_state.test_trade_symbol = test_symbol
                st.session_state.test_trade_quantity = abs(float(existing_test_position.qty))
        test_button_label = "Close RSI test trade" if st.session_state.test_trade_active else "Open RSI test trade"
        if st.button(test_button_label, type="primary", width="stretch", icon=":material/swap_vert:"):
            try:
                if st.session_state.test_trade_active:
                    test_order = submit_test_trade(
                        paper_client,
                        st.session_state.test_trade_symbol,
                        st.session_state.test_trade_quantity,
                        opening=False,
                    )
                    st.session_state.test_trade_active = False
                    st.session_state.test_trade_message = f"RSI test trade closed: {test_order.id}"
                else:
                    test_order = submit_test_trade(paper_client, test_symbol, test_quantity, opening=True)
                    st.session_state.test_trade_symbol = test_symbol
                    st.session_state.test_trade_quantity = test_quantity
                    st.session_state.test_trade_active = True
                    st.session_state.test_trade_message = f"RSI test trade opened: {test_order.id}"
                st.rerun()
            except Exception as error:
                st.error(f"RSI test trade failed: {error}")

        with st.form("paper_order_form", clear_on_submit=True):
            order_columns = st.columns([1.2, 1, 1])
            order_symbol = order_columns[0].text_input("Symbol", value=ticker, max_chars=10).strip().upper()
            order_side = order_columns[1].selectbox("Side", ["Buy", "Sell"])
            order_quantity = order_columns[2].number_input("Shares", min_value=1.0, value=1.0, step=1.0)
            confirm_order = st.checkbox("I understand this submits a PAPER market order.")
            submit_order = st.form_submit_button("Submit paper order", type="primary", icon=":material/send:")

        if submit_order:
            if not confirm_order:
                st.error("Confirm that this is a paper order before submitting.")
            elif not order_symbol:
                st.error("Enter a ticker symbol.")
            else:
                try:
                    submitted_order = submit_paper_market_order(paper_client, order_symbol, order_side, order_quantity)
                    st.success(f"Paper order submitted: {submitted_order.id}")
                except Exception as error:
                    st.error(f"Paper order failed: {error}")
    except Exception as error:
        st.warning(f"Alpaca paper trading is unavailable: {error}")

st.markdown("<div class='section-kicker'>01 / performance overview</div>", unsafe_allow_html=True)
st.subheader("Performance snapshot")
metric_rows = [
    [("Total Return", metrics["Total Return"], "percent"), ("CAGR", metrics["CAGR"], "percent"), ("Max Drawdown", metrics["Max Drawdown"], "percent"), ("Final Equity", metrics["Final Equity"], "money")],
    [("Sharpe Ratio", metrics["Sharpe"], "decimal"), ("Sortino Ratio", metrics["Sortino"], "decimal"), ("Win Rate", metrics["Win Rate"], "percent"), ("Trades", metrics["Trades"], "integer")],
]
for row in metric_rows:
    columns = st.columns(4)
    for column, (label, value, value_format) in zip(columns, row):
        with column:
            metric_card(label, value, value_format)

equity = results["equity"].dropna()
drawdown = equity / equity.cummax() - 1
benchmark = initial_cash * results["Close"] / results["Close"].iloc[0]
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy", line={"color": "#55d6be", "width": 2}))
fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name="Buy & hold", line={"color": "#8b9aa3", "width": 1, "dash": "dot"}))
fig.update_layout(height=390, margin={"l": 0, "r": 0, "t": 24, "b": 0}, template="plotly_dark", paper_bgcolor="#0b1015", plot_bgcolor="#0f171d", yaxis_title="Portfolio value ($)", legend={"orientation": "h", "y": 1.08})
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.markdown("<div class='section-kicker'>02 / risk profile</div>", unsafe_allow_html=True)
    st.subheader("Drawdown")
    drawdown_fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown * 100, fill="tozeroy", line={"color": "#ed6a5a"}, fillcolor="rgba(237,106,90,.18)"))
    drawdown_fig.update_layout(height=280, margin={"l": 0, "r": 0, "t": 8, "b": 0}, template="plotly_dark", paper_bgcolor="#0b1015", plot_bgcolor="#0f171d", yaxis_title="Drawdown (%)")
    st.plotly_chart(drawdown_fig, width="stretch")
with right:
    st.markdown("<div class='section-kicker'>03 / consistency</div>", unsafe_allow_html=True)
    st.subheader("Monthly returns")
    monthly = results["strategy_returns"].resample("ME").apply(lambda values: (1 + values.dropna()).prod() - 1)
    monthly_table = monthly.to_frame("Return").assign(Year=monthly.index.year, Month=monthly.index.month_name().str[:3])
    heatmap = monthly_table.pivot(index="Year", columns="Month", values="Return")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap = heatmap.reindex(columns=month_order)
    st.dataframe(heatmap.style.format("{:.2%}").background_gradient(cmap="RdYlGn", vmin=-0.1, vmax=0.1), width="stretch", height=280)

st.markdown("<div class='section-kicker'>04 / execution quality</div>", unsafe_allow_html=True)
st.subheader("Trade analysis")
trade_columns = st.columns(4)
trade_columns[0].metric("Profit factor", f"{metrics['Profit Factor']:.2f}")
trade_columns[1].metric("Average trade", f"{metrics['Avg Trade']:.2%}")
trade_columns[2].metric("Best trade", f"{trades['Return'].max():.2%}" if not trades.empty else "n/a")
trade_columns[3].metric("Worst trade", f"{trades['Return'].min():.2%}" if not trades.empty else "n/a")
if trades.empty:
    st.info("No completed trades were found for these parameters.")
else:
    display_trades = trades.copy()
    display_trades["Return"] = display_trades["Return"].map(lambda value: f"{value:.2%}")
    display_trades["Entry Price"] = display_trades["Entry Price"].map(lambda value: f"${value:,.2f}")
    display_trades["Exit Price"] = display_trades["Exit Price"].map(lambda value: f"${value:,.2f}")
    st.dataframe(display_trades.sort_values("Entry", ascending=False), width="stretch", hide_index=True)

st.markdown("<div class='section-kicker'>05 / audit trail</div>", unsafe_allow_html=True)
st.subheader("Signals and data")
signal_columns = [
    column
    for column in ["Close", "sma_fast", "sma_slow", "rsi", "position", "strategy_returns", "equity"]
    if column in results.columns
]
st.dataframe(results[signal_columns].tail(50).sort_index(ascending=False), width="stretch")
download_columns = results.reset_index()
st.download_button("Download full results CSV", download_columns.to_csv(index=False).encode("utf-8"), f"{ticker.lower()}_backtest.csv", "text/csv", icon=":material/download:")