# tests/journal/test_metrics_calculator.py
"""Tests for MetricsCalculator."""
from datetime import datetime

import pytest

from src.journal.metrics_calculator import MetricsCalculator
from src.journal.models import JournalEntry, TradingMetrics
from src.scoring.models import Direction, ScoreComponents


def make_score_components() -> ScoreComponents:
    """Create default score components for testing."""
    return ScoreComponents(
        sentiment_score=80.0,
        technical_score=75.0,
        sentiment_weight=0.5,
        technical_weight=0.5,
        confluence_bonus=0.1,
        credibility_multiplier=1.0,
        time_factor=1.0,
    )


def make_closed_entry(
    trade_id: str,
    symbol: str,
    pnl_dollars: float,
    pnl_percent: float,
    r_multiple: float,
    entry_price: float = 100.0,
    stop_loss: float = 98.0,
) -> JournalEntry:
    """Create a closed journal entry for testing."""
    return JournalEntry(
        trade_id=trade_id,
        symbol=symbol,
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 17, 9, 30, 0),
        entry_price=entry_price,
        entry_quantity=100,
        entry_reason="test_reason",
        entry_score=85.0,
        stop_loss=stop_loss,
        exit_time=datetime(2026, 1, 17, 10, 30, 0),
        exit_price=entry_price + (pnl_dollars / 100),
        exit_quantity=100,
        exit_reason="take_profit",
        pnl_dollars=pnl_dollars,
        pnl_percent=pnl_percent,
        r_multiple=r_multiple,
        market_conditions="bullish",
        score_components=make_score_components(),
        emotion_tag=None,
        notes=None,
    )


def make_open_entry(trade_id: str, symbol: str) -> JournalEntry:
    """Create an open journal entry for testing."""
    return JournalEntry(
        trade_id=trade_id,
        symbol=symbol,
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 17, 9, 30, 0),
        entry_price=100.0,
        entry_quantity=100,
        entry_reason="test_reason",
        entry_score=85.0,
        stop_loss=98.0,
        exit_time=None,
        exit_price=None,
        exit_quantity=0,
        exit_reason=None,
        pnl_dollars=0.0,
        pnl_percent=0.0,
        r_multiple=0.0,
        market_conditions="bullish",
        score_components=make_score_components(),
        emotion_tag=None,
        notes=None,
    )


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculate_with_no_entries(self) -> None:
        """calculate should return zero metrics when no entries provided."""
        calculator = MetricsCalculator()

        metrics = calculator.calculate(entries=[], period_days=30)

        assert metrics.period_days == 30
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.expectancy == 0.0
        assert metrics.total_pnl_dollars == 0.0
        assert metrics.best_trade is None
        assert metrics.worst_trade is None

    def test_calculate_win_rate(self) -> None:
        """calculate should compute correct win rate."""
        calculator = MetricsCalculator()
        entries = [
            make_closed_entry("1", "NVDA", pnl_dollars=100.0, pnl_percent=1.0, r_multiple=1.0),
            make_closed_entry("2", "AAPL", pnl_dollars=150.0, pnl_percent=1.5, r_multiple=1.5),
            make_closed_entry("3", "TSLA", pnl_dollars=-50.0, pnl_percent=-0.5, r_multiple=-0.5),
            make_closed_entry("4", "MSFT", pnl_dollars=80.0, pnl_percent=0.8, r_multiple=0.8),
        ]

        metrics = calculator.calculate(entries=entries, period_days=30)

        assert metrics.total_trades == 4
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.75

    def test_calculate_profit_factor(self) -> None:
        """calculate should compute correct profit factor."""
        calculator = MetricsCalculator()
        # Gross profit: 100 + 200 = 300
        # Gross loss: 50 + 100 = 150
        # Profit factor: 300 / 150 = 2.0
        entries = [
            make_closed_entry("1", "NVDA", pnl_dollars=100.0, pnl_percent=1.0, r_multiple=1.0),
            make_closed_entry("2", "AAPL", pnl_dollars=200.0, pnl_percent=2.0, r_multiple=2.0),
            make_closed_entry("3", "TSLA", pnl_dollars=-50.0, pnl_percent=-0.5, r_multiple=-0.5),
            make_closed_entry("4", "MSFT", pnl_dollars=-100.0, pnl_percent=-1.0, r_multiple=-1.0),
        ]

        metrics = calculator.calculate(entries=entries, period_days=30)

        assert metrics.profit_factor == 2.0

    def test_calculate_expectancy(self) -> None:
        """calculate should compute correct expectancy."""
        calculator = MetricsCalculator()
        # 3 wins: r_multiple = 1.0, 1.5, 2.0 -> avg_win_r = 1.5
        # 1 loss: r_multiple = -1.0 -> avg_loss_r = 1.0
        # win_rate = 0.75, loss_rate = 0.25
        # expectancy = (0.75 * 1.5) - (0.25 * 1.0) = 1.125 - 0.25 = 0.875
        entries = [
            make_closed_entry("1", "NVDA", pnl_dollars=100.0, pnl_percent=1.0, r_multiple=1.0),
            make_closed_entry("2", "AAPL", pnl_dollars=150.0, pnl_percent=1.5, r_multiple=1.5),
            make_closed_entry("3", "TSLA", pnl_dollars=-100.0, pnl_percent=-1.0, r_multiple=-1.0),
            make_closed_entry("4", "MSFT", pnl_dollars=200.0, pnl_percent=2.0, r_multiple=2.0),
        ]

        metrics = calculator.calculate(entries=entries, period_days=30)

        assert metrics.expectancy == pytest.approx(0.875, rel=0.01)

    def test_calculate_total_pnl(self) -> None:
        """calculate should compute correct total PnL."""
        calculator = MetricsCalculator()
        entries = [
            make_closed_entry("1", "NVDA", pnl_dollars=100.0, pnl_percent=1.0, r_multiple=1.0),
            make_closed_entry("2", "AAPL", pnl_dollars=200.0, pnl_percent=2.0, r_multiple=2.0),
            make_closed_entry("3", "TSLA", pnl_dollars=-50.0, pnl_percent=-0.5, r_multiple=-0.5),
        ]

        metrics = calculator.calculate(entries=entries, period_days=30)

        assert metrics.total_pnl_dollars == 250.0
        assert metrics.total_pnl_percent == pytest.approx(2.5, rel=0.01)

    def test_calculate_best_worst_trades(self) -> None:
        """calculate should identify best and worst trades by pnl_dollars."""
        calculator = MetricsCalculator()
        best_entry = make_closed_entry("2", "AAPL", pnl_dollars=500.0, pnl_percent=5.0, r_multiple=5.0)
        worst_entry = make_closed_entry("3", "TSLA", pnl_dollars=-200.0, pnl_percent=-2.0, r_multiple=-2.0)
        entries = [
            make_closed_entry("1", "NVDA", pnl_dollars=100.0, pnl_percent=1.0, r_multiple=1.0),
            best_entry,
            worst_entry,
            make_closed_entry("4", "MSFT", pnl_dollars=50.0, pnl_percent=0.5, r_multiple=0.5),
        ]

        metrics = calculator.calculate(entries=entries, period_days=30)

        assert metrics.best_trade is not None
        assert metrics.best_trade.trade_id == "2"
        assert metrics.best_trade.pnl_dollars == 500.0
        assert metrics.worst_trade is not None
        assert metrics.worst_trade.trade_id == "3"
        assert metrics.worst_trade.pnl_dollars == -200.0

    def test_calculate_avg_win_loss(self) -> None:
        """calculate should compute correct average win and loss amounts."""
        calculator = MetricsCalculator()
        # Wins: 100, 200, 300 -> avg = 200
        # Losses: -50, -150 -> avg = -100
        entries = [
            make_closed_entry("1", "NVDA", pnl_dollars=100.0, pnl_percent=1.0, r_multiple=1.0),
            make_closed_entry("2", "AAPL", pnl_dollars=200.0, pnl_percent=2.0, r_multiple=2.0),
            make_closed_entry("3", "TSLA", pnl_dollars=-50.0, pnl_percent=-0.5, r_multiple=-0.5),
            make_closed_entry("4", "MSFT", pnl_dollars=300.0, pnl_percent=3.0, r_multiple=3.0),
            make_closed_entry("5", "GOOG", pnl_dollars=-150.0, pnl_percent=-1.5, r_multiple=-1.5),
        ]

        metrics = calculator.calculate(entries=entries, period_days=30)

        assert metrics.avg_win_dollars == 200.0
        assert metrics.avg_loss_dollars == -100.0
        assert metrics.avg_win_r == 2.0
        assert metrics.avg_loss_r == 1.0

    def test_ignores_open_trades(self) -> None:
        """calculate should only include closed trades."""
        calculator = MetricsCalculator()
        entries = [
            make_closed_entry("1", "NVDA", pnl_dollars=100.0, pnl_percent=1.0, r_multiple=1.0),
            make_open_entry("2", "AAPL"),
            make_closed_entry("3", "TSLA", pnl_dollars=-50.0, pnl_percent=-0.5, r_multiple=-0.5),
            make_open_entry("4", "MSFT"),
        ]

        metrics = calculator.calculate(entries=entries, period_days=30)

        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.total_pnl_dollars == 50.0
