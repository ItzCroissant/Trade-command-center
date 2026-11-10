import performance
import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timezone

# =========================================================
# TRADING COMMAND CENTER
# Version 2.0 | Paper Trading Default | Live Trading Gated
# =========================================================

st.set_page_config(
    page_title="Trading Command Center",
    page_icon="📈",
    layout="wide"
)

# ---------------- CONFIGURATION ----------------

STARTING_CAPITAL = 3.00
LIVE_TRADING_ENABLED = False

COINBASE_API = "https://api.exchange.coinbase.com"

# ---------------- SESSION STATE ----------------

defaults = {
    "kalshi_cash": 3.00,
    "invest_cash": 3.00,
    "kalshi_trades": [],
    "invest_trades": [],
    "prediction_history": [],
    "live_approved": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {background-color:#0b1220;color:white;}
section[data-testid="stSidebar"] {background:#111c30;}
div[data-testid="stMetric"] {
    background:#17243a;
    padding:15px;
    border-radius:12px;
    border:1px solid #263750;
}
h1,h2,h3 {color:#ffffff;}
</style>
""", unsafe_allow_html=True)

# ---------------- MARKET DATA ----------------

@st.cache_data(ttl=15)
def get_crypto_price(symbol):
    try:
        response = requests.get(
            f"{COINBASE_API}/products/{symbol}/ticker",
            timeout=10
        )
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_candles(symbol, granularity=900):
    try:
        response = requests.get(
            f"{COINBASE_API}/products/{symbol}/candles",
            params={"granularity": granularity},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(
            data,
            columns=[
                "time", "low", "high",
                "open", "close", "volume"
            ]
        )

        df["time"] = pd.to_datetime(
            df["time"], unit="s", utc=True
        )

        df = df.sort_values("time").reset_index(drop=True)

        for col in ["low", "high", "open", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        return df

    except Exception:
        return pd.DataFrame()


# ---------------- INDICATORS ----------------

def calculate_indicators(df):

    if df.empty or len(df) < 30:
        return None

    data = df.copy()

    data["EMA9"] = data["close"].ewm(span=9).mean()
    data["EMA21"] = data["close"].ewm(span=21).mean()
    data["EMA50"] = data["close"].ewm(span=50).mean()

    delta = data["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI"] = data["RSI"].fillna(50)

    data["MACD"] = (
        data["close"].ewm(span=12).mean()
        - data["close"].ewm(span=26).mean()
    )

    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9).mean()

    data["VOL_AVG"] = data["volume"].rolling(20).mean()

    data["ATR"] = (
        data["high"] - data["low"]
    ).rolling(14).mean()

    return data


# ---------------- PREDICTION ENGINE ----------------

def predict_direction(df):

    data = calculate_indicators(df)

    if data is None:
        return None

    current = data.iloc[-1]

    score = 0
    reasons = []

    # Trend
    if current["EMA9"] > current["EMA21"]:
        score += 1
        reasons.append("Short-term EMA trend bullish")
    else:
        score -= 1
        reasons.append("Short-term EMA trend bearish")

    # Long-term trend
    if current["EMA21"] > current["EMA50"]:
        score += 1
        reasons.append("Longer-term EMA trend bullish")
    else:
        score -= 1
        reasons.append("Longer-term EMA trend bearish")

    # RSI
    if current["RSI"] > 55:
        score += 1
        reasons.append("RSI momentum bullish")
    elif current["RSI"] < 45:
        score -= 1
        reasons.append("RSI momentum bearish")

    # MACD
    if current["MACD"] > current["MACD_SIGNAL"]:
        score += 1
        reasons.append("MACD bullish")
    else:
        score -= 1
        reasons.append("MACD bearish")

    # Volume confirmation
    if pd.notna(current["VOL_AVG"]):
        if current["volume"] > current["VOL_AVG"]:
            reasons.append("Volume above recent average")

    # Signal
    if score >= 3:
        signal = "UP"
    elif score <= -3:
        signal = "DOWN"
    else:
        signal = "WAIT"

    # This is a heuristic score, NOT a calibrated probability.
    confidence_score = min(abs(score) / 4, 1.0)

    return {
        "signal": signal,
        "score": score,
        "confidence_score": confidence_score,
        "rsi": float(current["RSI"]),
        "macd": float(current["MACD"]),
        "reasons": reasons,
        "price": float(current["close"]),
        "timestamp": str(current["time"])
    }


# ---------------- PAPER TRADING ----------------

def open_paper_trade(engine, symbol, direction, stake, price):

    if price is None or not np.isfinite(price) or price <= 0:
        return False, "Price unavailable."

    cash_key = (
        "kalshi_cash" if engine == "kalshi"
        else "invest_cash"
    )

    trade_key = (
        "kalshi_trades" if engine == "kalshi"
        else "invest_trades"
    )

    cash = st.session_state[cash_key]

    if stake <= 0 or stake > cash:
        return False, "Insufficient simulated funds."

    trade = {
        "symbol": symbol,
        "direction": direction,
        "stake": float(stake),
        "entry": float(price),
        "opened": datetime.now(timezone.utc).isoformat()
    }

    st.session_state[cash_key] -= stake
    st.session_state[trade_key].append(trade)

    return True, "Paper trade opened."


def close_paper_trade(engine, index, current_price):

    cash_key = (
        "kalshi_cash" if engine == "kalshi"
        else "invest_cash"
    )

    trade_key = (
        "kalshi_trades" if engine == "kalshi"
        else "invest_trades"
    )

    trades = st.session_state[trade_key]

    if index >= len(trades):
        return

    if current_price is None or current_price <= 0:
        return

    trade = trades[index]

    movement = (
        current_price - trade["entry"]
    ) / trade["entry"]

    if trade["direction"] == "DOWN":
        movement = -movement

    pnl = trade["stake"] * movement
    pnl = max(-trade["stake"], pnl)

    returned = trade["stake"] + pnl

    st.session_state[cash_key] += returned

    trades.pop(index)


# ---------------- HISTORICAL BACKTEST ----------------

def backtest(df, threshold=3):

    data = calculate_indicators(df)

    if data is None or len(data) < 100:
        return None

    wins = 0
    losses = 0
    total = 0

    # Only use indicators available at the prediction candle.
    # Evaluate direction against the next candle's close.
    for i in range(50, len(data) - 1):

        row = data.iloc[i]

        score = 0

        if row["EMA9"] > row["EMA21"]:
            score += 1
        else:
            score -= 1

        if row["EMA21"] > row["EMA50"]:
            score += 1
        else:
            score -= 1

        if row["RSI"] > 55:
            score += 1
        elif row["RSI"] < 45:
            score -= 1

        if row["MACD"] > row["MACD_SIGNAL"]:
            score += 1
        else:
            score -= 1

        if score >= threshold:
            direction = 1
        elif score <= -threshold:
            direction = -1
        else:
            continue

        future_change = (
            data.iloc[i + 1]["close"] - row["close"]
        )

        total += 1

        if future_change * direction > 0:
            wins += 1
        else:
            losses += 1

    if total == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "accuracy": 0
        }

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "accuracy": wins / total * 100
    }


# ---------------- HEADER ----------------

st.title("📈 TRADING COMMAND CENTER")

st.caption(
    "Multi-market research | AI-ready prediction architecture | "
    "Paper trading default"
)

left, middle, right = st.columns(3)

left.metric(
    "Kalshi simulated balance",
    f"${st.session_state.kalshi_cash:.2f}"
)

middle.metric(
    "Investing simulated balance",
    f"${st.session_state.invest_cash:.2f}"
)

right.metric(
    "Combined available cash",
    f"${st.session_state.kalshi_cash + st.session_state.invest_cash:.2f}"
)

# ---------------- NAVIGATION ----------------

tabs = st.tabs([
    "🏠 Overview",
    "₿ Kalshi Engine",
    "📊 Investing Engine",
    "👁 Trade Monitor",
    "🧠 Strategy Lab",
    "⚙️ Settings"
])


# =========================================================
# OVERVIEW
# =========================================================

with tabs[0]:

    st.header("Market Overview")

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

    columns = st.columns(3)

    for column, symbol in zip(columns, symbols):

        price = get_crypto_price(symbol)

        with column:

            if price is not None:
                st.metric(symbol, f"${price:,.2f}")
            else:
                st.metric(symbol, "Unavailable")

    st.info(
        "BTC spot prices are market reference data, not "
        "Kalshi contract prices or guaranteed execution prices."
    )

    st.subheader("System status")

    st.success("Paper trading: ACTIVE")
    st.warning("Real-money trading: DISABLED")
    st.info("Prediction engine: Experimental baseline")


# =========================================================
# KALSHI ENGINE
# =========================================================

with tabs[1]:

    st.header("Bitcoin Prediction Engine")

    timeframe = st.selectbox(
        "Prediction timeframe",
        ["15-minute", "1-hour", "4-hour"]
    )

    granularity = {
        "15-minute": 900,
        "1-hour": 3600,
        "4-hour": 3600
    }[timeframe]

    btc_price = get_crypto_price("BTC-USD")

    btc_data = get_candles("BTC-USD", granularity)

    if btc_price is not None:
        st.metric("BTC Spot Price", f"${btc_price:,.2f}")

    prediction = predict_direction(btc_data)

    if prediction:

        signal = prediction["signal"]

        if signal == "UP":
            st.success("PREDICTION SIGNAL: UP")

        elif signal == "DOWN":
            st.error("PREDICTION SIGNAL: DOWN")

        else:
            st.warning("PREDICTION SIGNAL: WAIT")

        a, b, c = st.columns(3)

        a.metric("RSI", f"{prediction['rsi']:.2f}")
        b.metric("Indicator score", prediction["score"])
        c.metric(
            "Heuristic strength",
            f"{prediction['confidence_score'] * 100:.0f}%"
        )

        st.caption(
            "Heuristic strength is not a measured probability or "
            "historical accuracy."
        )

        st.subheader("Signal explanation")

        for reason in prediction["reasons"]:
            st.write("• " + reason)

        if st.button("Record prediction"):

            st.session_state.prediction_history.append({
                "time": datetime.now(timezone.utc).isoformat(),
                "symbol": "BTC-USD",
                "timeframe": timeframe,
                "signal": signal,
                "price": prediction["price"]
            })

            st.success("Prediction recorded.")

    else:

        st.warning("Waiting for sufficient BTC market data.")

    st.divider()

    st.subheader("Open BTC paper trade")

    direction = st.radio(
        "Trade direction",
        ["UP", "DOWN"],
        horizontal=True,
        key="btc_direction"
    )

    stake = st.number_input(
        "Stake ($)",
        min_value=0.01,
        max_value=max(0.01, st.session_state.kalshi_cash),
        value=min(0.50, st.session_state.kalshi_cash),
        step=0.01,
        key="btc_stake"
    )

    if st.button("Execute BTC paper trade"):

        success, message = open_paper_trade(
            "kalshi",
            "BTC-USD",
            direction,
            stake,
            btc_price
        )

        if success:
            st.success(message)
        else:
            st.error(message)


# =========================================================
# INVESTING ENGINE
# =========================================================

with tabs[2]:

    st.header("Investing Engine")

    st.write(
        "Stocks, ETFs and cryptocurrency research. "
        "Overnight strategy development."
    )

    asset = st.selectbox(
        "Select asset",
        ["BTC-USD", "ETH-USD", "SOL-USD"]
    )

    price = get_crypto_price(asset)

    candles = get_candles(asset, 3600)

    result = predict_direction(candles)

    if price is not None:
        st.metric("Current spot price", f"${price:,.2f}")

    if result:

        st.subheader("Technical research")

        st.write(f"Signal: **{result['signal']}**")
        st.write(f"RSI: {result['rsi']:.2f}")

        for reason in result["reasons"]:
            st.write("• " + reason)

    st.divider()

    st.subheader("Investing paper trade")

    invest_direction = st.radio(
        "Direction",
        ["UP", "DOWN"],
        horizontal=True,
        key="invest_direction"
    )

    invest_stake = st.number_input(
        "Investment amount ($)",
        min_value=0.01,
        max_value=max(0.01, st.session_state.invest_cash),
        value=min(0.50, st.session_state.invest_cash),
        step=0.01,
        key="invest_stake"
    )

    if st.button("Execute investing paper trade"):

        success, message = open_paper_trade(
            "invest",
            asset,
            invest_direction,
            invest_stake,
            price
        )

        if success:
            st.success(message)
        else:
            st.error(message)


# =========================================================
# TRADE MONITOR
# =========================================================

with tabs[3]:

    st.header("Live Paper Trade Monitor")

    if st.button("Refresh trade prices"):
        st.cache_data.clear()
        st.rerun()

    for engine, trade_key in [
        ("kalshi", "kalshi_trades"),
        ("invest", "invest_trades")
    ]:

        st.subheader(engine.upper())

        trades = st.session_state[trade_key]

        if not trades:
            st.info("No open paper trades.")
            continue

        for index, trade in enumerate(list(trades)):

            price = get_crypto_price(trade["symbol"])

            with st.container(border=True):

                st.write(
                    f"**{trade['symbol']} — {trade['direction']}**"
                )

                st.write(f"Entry: ${trade['entry']:,.4f}")
                st.write(f"Stake: ${trade['stake']:.2f}")

                if price is not None:

                    change = (
                        price - trade["entry"]
                    ) / trade["entry"]

                    if trade["direction"] == "DOWN":
                        change *= -1

                    pnl = trade["stake"] * change

                    st.metric(
                        "Estimated paper P/L",
                        f"${pnl:+.4f}",
                        f"{change * 100:+.3f}%"
                    )

                    if st.button(
                        "Close paper trade",
                        key=f"close_{engine}_{index}"
                    ):

                        close_paper_trade(
                            engine,
                            index,
                            price
                        )

                        st.rerun()

                else:
                    st.warning("Unable to retrieve current price.")


# =========================================================
# STRATEGY LAB
# =========================================================

with tabs[4]:

    st.header("Strategy Research Laboratory")

    st.write(
        "Test technical signals against historical candle data. "
        "Historical performance does not guarantee future results."
    )

    test_timeframe = st.selectbox(
        "Backtest timeframe",
        ["15-minute", "1-hour"],
        key="backtest_timeframe"
    )

    test_granularity = (
        900 if test_timeframe == "15-minute" else 3600
    )

    historical = get_candles(
        "BTC-USD",
        test_granularity
    )

    st.write(
        f"Available candles in current data response: {len(historical)}"
    )

    if st.button("Run BTC baseline backtest"):

        result = backtest(historical)

        if result:

            a, b, c = st.columns(3)

            a.metric("Historical signals", result["trades"])
            b.metric("Wins", result["wins"])
            c.metric("Directional accuracy", f"{result['accuracy']:.2f}%")

            st.write(f"Losses: {result['losses']}")

            st.warning(
                "This is a short-sample directional baseline, "
                "not a five-year test or a profitability estimate. "
                "Fees, spreads, slippage and execution are not included."
            )

        else:

            st.warning("Insufficient historical data.")

    st.divider()

    st.subheader("Prediction history")

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        st.dataframe(history_df, use_container_width=True)

    else:

        st.info("No predictions recorded yet.")


# =========================================================
# SETTINGS
# =========================================================

with tabs[5]:

    st.header("System Settings")

    st.write("Trading mode: PAPER")

    st.write("Live execution: Disabled")

    st.checkbox(
        "I understand that live trading is not implemented.",
        value=False,
        disabled=True
    )

    st.warning(
        "No live Kalshi orders or brokerage orders can be placed "
        "by this version. A future live-trading module will require "
        "separate implementation, secure credentials, and explicit approval."
    )

    st.subheader("Virtual account balances")

    st.write(
        f"Kalshi: ${st.session_state.kalshi_cash:.2f}"
    )

    st.write(
        f"Investing: ${st.session_state.invest_cash:.2f}"
    )

    if st.button("Reset both paper accounts to $3"):

        st.session_state.kalshi_cash = STARTING_CAPITAL
        st.session_state.invest_cash = STARTING_CAPITAL

        st.session_state.kalshi_trades = []
        st.session_state.invest_trades = []

        st.success("Paper accounts reset.")

        st.rerun()


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Trading Command Center v2.0 | Experimental research software | "
    "Paper trading only | No guaranteed returns"
)
