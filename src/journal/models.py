# src/journal/models.py
"""Data models for the trading journal."""
from dataclasses import dataclass
from datetime import datetime

from src.scoring.models import Direction, ScoreComponents


@dataclass
class JournalEntry:
    """A single trade entry in the journal."""

    trade_id: str
    symbol: str
    direction: Direction

    # Entry details
    entry_time: datetime
    entry_price: float
    entry_quantity: int
    entry_reason: str
    entry_score: float
    stop_loss: float

    # Exit details
    exit_time: datetime | None
    exit_price: float | None
    exit_quantity: int
    exit_reason: str | None

    # Results
    pnl_dollars: float
    pnl_percent: float
    r_multiple: float

    # Context
    market_conditions: str
    score_components: ScoreComponents

    # Manual tags
    emotion_tag: str | None
    notes: str | None

    @property
    def is_open(self) -> bool:
        """Check if the trade is still open."""
        return self.exit_time is None


@dataclass
class TradingMetrics:
    """Calculated trading performance metrics."""

    period_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int

    win_rate: float
    profit_factor: float
    expectancy: float

    avg_win_dollars: float
    avg_loss_dollars: float
    avg_win_r: float
    avg_loss_r: float

    total_pnl_dollars: float
    total_pnl_percent: float
    max_drawdown_percent: float
    sharpe_ratio: float

    best_trade: JournalEntry | None
    worst_trade: JournalEntry | None


@dataclass
class PatternAnalysis:
    """Analysis of trading patterns."""

    best_hour: int
    worst_hour: int
    best_day_of_week: int

    best_symbols: list[tuple[str, float]]
    worst_symbols: list[tuple[str, float]]

    best_setups: list[tuple[str, float]]
    worst_setups: list[tuple[str, float]]

    avg_winner_duration_minutes: float
    avg_loser_duration_minutes: float
