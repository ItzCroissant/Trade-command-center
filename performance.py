
import sqlite3
import uuid
from datetime import datetime, timezone

import pandas as pd

DB_PATH = "trades.db"


def connect():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                market TEXT NOT NULL,
                ticker TEXT,
                direction TEXT NOT NULL,
                strategy TEXT NOT NULL,
                timeframe TEXT,
                entry_price REAL NOT NULL,
                contracts INTEGER NOT NULL,
                fees REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN',
                outcome TEXT,
                exit_value REAL,
                realized_pnl REAL,
                expected_probability REAL,
                created_at TEXT NOT NULL,
                settled_at TEXT
            )
        """)
        conn.commit()


def record_trade(
    market,
    direction,
    strategy,
    entry_price,
    contracts,
    fees=0,
    ticker="",
    timeframe="",
    expected_probability=None
):
    if direction.upper() not in ("YES", "NO"):
        raise ValueError("Direction must be YES or NO.")

    if not 0 < entry_price < 1:
        raise ValueError("Entry price must be between 0 and 1.")

    if contracts < 1:
        raise ValueError("Contracts must be at least 1.")

    if fees < 0:
        raise ValueError("Fees cannot be negative.")

    if expected_probability is not None:
        if not 0 <= expected_probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")

    trade_id = str(uuid.uuid4())

    with connect() as conn:
        conn.execute("""
            INSERT INTO trades (
                trade_id, market, ticker, direction,
                strategy, timeframe, entry_price,
                contracts, fees, expected_probability,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, market, ticker, direction.upper(),
            strategy, timeframe, entry_price,
            contracts, fees, expected_probability,
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()

    return trade_id


def settle_trade(trade_id, outcome, exit_value=None):
    outcome = outcome.upper()

    if outcome not in ("YES", "NO"):
        raise ValueError("Outcome must be YES or NO.")

    if exit_value is not None and not 0 <= exit_value <= 1:
        raise ValueError("Exit value must be between 0 and 1.")

    with connect() as conn:
        trade = conn.execute("""
            SELECT direction, entry_price, contracts, fees, status
            FROM trades WHERE trade_id = ?
        """, (trade_id,)).fetchone()

        if trade is None:
            raise ValueError("Trade not found.")

        direction, entry, contracts, fees, status = trade

        if status != "OPEN":
            raise ValueError("Trade has already been settled.")

        if exit_value is not None:
            proceeds = exit_value * contracts
        else:
            proceeds = contracts if direction == outcome else 0

        cost = entry * contracts
        pnl = proceeds - cost - fees

        conn.execute("""
            UPDATE trades
            SET status = 'SETTLED',
                outcome = ?,
                exit_value = ?,
                realized_pnl = ?,
                settled_at = ?
            WHERE trade_id = ?
        """, (
            outcome, exit_value, pnl,
            datetime.now(timezone.utc).isoformat(),
            trade_id
        ))
        conn.commit()

    return pnl


def get_trades():
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM trades ORDER BY created_at DESC",
            conn
        )


def calculate_metrics(starting_bankroll=3.00):
    df = get_trades()

    if df.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "realized_pnl": 0,
            "open_trades": 0,
            "bankroll": starting_bankroll,
            "max_drawdown": 0
        }

    settled = df[df["status"] == "SETTLED"].copy()
    open_trades = df[df["status"] == "OPEN"]

    wins = int((settled["realized_pnl"] > 0).sum())
    losses = int((settled["realized_pnl"] <= 0).sum())

    pnl = float(settled["realized_pnl"].sum())

    equity_changes = settled.sort_values(
        "settled_at"
    )["realized_pnl"]

    equity = starting_bankroll + equity_changes.cumsum()
    peak = equity.cummax()

    drawdowns = peak - equity
    max_drawdown = (
        float(drawdowns.max()) if not drawdowns.empty else 0
    )

    total = len(settled)

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total * 100) if total else 0,
        "realized_pnl": pnl,
        "open_trades": len(open_trades),
        "bankroll": starting_bankroll + pnl,
        "max_drawdown": max_drawdown
    }


def strategy_performance():
    df = get_trades()
    settled = df[df["status"] == "SETTLED"].copy()

    if settled.empty:
        return pd.DataFrame()

    return settled.groupby("strategy").agg(
        trades=("trade_id", "count"),
        average_pnl=("realized_pnl", "mean"),
        total_pnl=("realized_pnl", "sum"),
        wins=("realized_pnl", lambda x: (x > 0).sum())
    ).reset_index()


initialize_database()
