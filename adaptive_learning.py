
import sqlite3
import math
import json
from datetime import datetime, timezone

DB_NAME = "trades.db"
MIN_PREDICTIONS = 100


def connect():
    return sqlite3.connect(DB_NAME)


def initialize_learning():
    """Create prediction history and model version tables."""
    with connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT NOT NULL,
                ticker TEXT,
                direction TEXT NOT NULL,
                probability REAL NOT NULL,
                strategy TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                model_version TEXT NOT NULL,
                decision TEXT DEFAULT 'WAIT',
                features TEXT,
                actual_outcome TEXT,
                created_at TEXT NOT NULL,
                settled_at TEXT
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                weights TEXT NOT NULL,
                status TEXT DEFAULT 'TESTING',
                created_at TEXT NOT NULL
            )
        """)


def record_prediction(
    market,
    direction,
    probability,
    strategy,
    timeframe,
    model_version="1.0",
    ticker=None,
    decision="WAIT",
    features=None
):
    """Save a prediction before the market settles."""

    direction = direction.upper()

    if direction not in ("UP", "DOWN"):
        raise ValueError("Direction must be UP or DOWN.")

    if not 0 < probability < 1:
        raise ValueError("Probability must be between 0 and 1.")

    with connect() as db:
        cursor = db.execute("""
            INSERT INTO predictions (
                market, ticker, direction, probability,
                strategy, timeframe, model_version,
                decision, features, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market,
            ticker,
            direction,
            probability,
            strategy,
            timeframe,
            model_version,
            decision,
            json.dumps(features or {}),
            datetime.now(timezone.utc).isoformat()
        ))

        return cursor.lastrowid


def settle_prediction(prediction_id, actual_outcome):
    """Record the official market outcome."""

    actual_outcome = actual_outcome.upper()

    if actual_outcome not in ("UP", "DOWN"):
        raise ValueError("Outcome must be UP or DOWN.")

    with connect() as db:
        db.execute("""
            UPDATE predictions
            SET actual_outcome = ?, settled_at = ?
            WHERE id = ? AND actual_outcome IS NULL
        """, (
            actual_outcome,
            datetime.now(timezone.utc).isoformat(),
            prediction_id
        ))


def evaluate_strategy(strategy, timeframe=None):
    """Evaluate a strategy using only settled predictions."""

    with connect() as db:
        if timeframe:
            rows = db.execute("""
                SELECT direction, probability, actual_outcome
                FROM predictions
                WHERE strategy = ?
                AND timeframe = ?
                AND actual_outcome IS NOT NULL
            """, (strategy, timeframe)).fetchall()
        else:
            rows = db.execute("""
                SELECT direction, probability, actual_outcome
                FROM predictions
                WHERE strategy = ?
                AND actual_outcome IS NOT NULL
            """, (strategy,)).fetchall()

    total = len(rows)

    if total == 0:
        return {
            "strategy": strategy,
            "resolved_predictions": 0,
            "status": "NOT ENOUGH DATA"
        }

    correct = sum(
        1 for direction, probability, outcome in rows
        if direction == outcome
    )

    brier_scores = []
    log_losses = []

    for direction, probability, outcome in rows:
        predicted_up = (
            probability if direction == "UP"
            else 1 - probability
        )

        actual_up = 1 if outcome == "UP" else 0

        brier_scores.append((predicted_up - actual_up) ** 2)

        p = min(max(predicted_up, 0.000001), 0.999999)

        log_losses.append(
            -(actual_up * math.log(p)
              + (1 - actual_up) * math.log(1 - p))
        )

    return {
        "strategy": strategy,
        "resolved_predictions": total,
        "accuracy": round(correct / total * 100, 2),
        "brier_score": round(sum(brier_scores) / total, 4),
        "log_loss": round(sum(log_losses) / total, 4),
        "status": (
            "READY FOR REVIEW"
            if total >= MIN_PREDICTIONS
            else "COLLECTING DATA"
        )
    }


def learning_status():
    """Show whether the engine has enough evidence to evaluate."""

    with connect() as db:
        total = db.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE actual_outcome IS NOT NULL
        """).fetchone()[0]

        pending = db.execute("""
            SELECT COUNT(*) FROM predictions
            WHERE actual_outcome IS NULL
        """).fetchone()[0]

    return {
        "resolved_predictions": total,
        "pending_predictions": pending,
        "minimum_required": MIN_PREDICTIONS,
        "ready_for_review": total >= MIN_PREDICTIONS,
        "automatic_weight_changes": False,
        "live_trading_changes": False
    }


def save_model_candidate(version, weights):
    """Save proposed weights for human review. Never activates them."""

    with connect() as db:
        db.execute("""
            INSERT INTO model_versions (
                version, weights, status, created_at
            )
            VALUES (?, ?, 'PENDING REVIEW', ?)
        """, (
            version,
            json.dumps(weights),
            datetime.now(timezone.utc).isoformat()
        ))


# Initialize database tables when this module is imported.
initialize_learning()

def get_strategy_names():
    """Return all strategies with recorded predictions."""

    with connect() as db:
        rows = db.execute("""
            SELECT DISTINCT strategy
            FROM predictions
            ORDER BY strategy
        """).fetchall()

    return [row[0] for row in rows]
