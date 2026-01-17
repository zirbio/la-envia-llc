# tests/journal/test_models.py
"""Tests for journal models."""
import pytest
from datetime import datetime

from src.journal.models import JournalEntry, TradingMetrics, PatternAnalysis
from src.scoring.models import Direction, ScoreComponents


class TestJournalEntry:
    """Tests for JournalEntry dataclass."""

    def test_create_entry_with_required_fields(self):
        """JournalEntry should store all trade data."""
        components = ScoreComponents(
            sentiment_score=80.0,
            technical_score=75.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )
        entry = JournalEntry(
            trade_id="2026-01-17-NVDA-001",
            symbol="NVDA",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 17, 9, 33, 0),
            entry_price=140.76,
            entry_quantity=322,
            entry_reason="unusual_whales_sweep_alert",
            entry_score=88.0,
            stop_loss=139.20,
            exit_time=None,
            exit_price=None,
            exit_quantity=0,
            exit_reason=None,
            pnl_dollars=0.0,
            pnl_percent=0.0,
            r_multiple=0.0,
            market_conditions="bullish_trend_high_volume",
            score_components=components,
            emotion_tag=None,
            notes=None,
        )

        assert entry.trade_id == "2026-01-17-NVDA-001"
        assert entry.symbol == "NVDA"
        assert entry.direction == Direction.LONG
        assert entry.entry_price == 140.76
        assert entry.entry_quantity == 322
        assert entry.exit_time is None

    def test_entry_is_open_when_no_exit(self):
        """is_open should return True when exit_time is None."""
        components = ScoreComponents(
            sentiment_score=80.0,
            technical_score=75.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )
        entry = JournalEntry(
            trade_id="2026-01-17-NVDA-001",
            symbol="NVDA",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 17, 9, 33, 0),
            entry_price=140.76,
            entry_quantity=322,
            entry_reason="test",
            entry_score=88.0,
            stop_loss=139.20,
            exit_time=None,
            exit_price=None,
            exit_quantity=0,
            exit_reason=None,
            pnl_dollars=0.0,
            pnl_percent=0.0,
            r_multiple=0.0,
            market_conditions="test",
            score_components=components,
            emotion_tag=None,
            notes=None,
        )

        assert entry.is_open is True

    def test_entry_is_closed_when_has_exit(self):
        """is_open should return False when exit_time is set."""
        components = ScoreComponents(
            sentiment_score=80.0,
            technical_score=75.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )
        entry = JournalEntry(
            trade_id="2026-01-17-NVDA-001",
            symbol="NVDA",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 17, 9, 33, 0),
            entry_price=140.76,
            entry_quantity=322,
            entry_reason="test",
            entry_score=88.0,
            stop_loss=139.20,
            exit_time=datetime(2026, 1, 17, 11, 45, 0),
            exit_price=142.38,
            exit_quantity=322,
            exit_reason="trailing_stop",
            pnl_dollars=521.44,
            pnl_percent=1.15,
            r_multiple=1.33,
            market_conditions="test",
            score_components=components,
            emotion_tag=None,
            notes=None,
        )

        assert entry.is_open is False


class TestTradingMetrics:
    """Tests for TradingMetrics dataclass."""

    def test_create_metrics(self):
        """TradingMetrics should store all calculated metrics."""
        metrics = TradingMetrics(
            period_days=30,
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            win_rate=0.6,
            profit_factor=1.8,
            expectancy=0.5,
            avg_win_dollars=150.0,
            avg_loss_dollars=100.0,
            avg_win_r=1.5,
            avg_loss_r=1.0,
            total_pnl_dollars=1000.0,
            total_pnl_percent=2.0,
            max_drawdown_percent=5.0,
            sharpe_ratio=1.2,
            best_trade=None,
            worst_trade=None,
        )

        assert metrics.win_rate == 0.6
        assert metrics.profit_factor == 1.8
        assert metrics.total_trades == 20


class TestPatternAnalysis:
    """Tests for PatternAnalysis dataclass."""

    def test_create_pattern_analysis(self):
        """PatternAnalysis should store all pattern data."""
        analysis = PatternAnalysis(
            best_hour=10,
            worst_hour=14,
            best_day_of_week=1,
            best_symbols=[("NVDA", 250.0), ("AAPL", 150.0)],
            worst_symbols=[("TSLA", -100.0)],
            best_setups=[("unusual_whales", 200.0)],
            worst_setups=[("fomo_entry", -150.0)],
            avg_winner_duration_minutes=45.0,
            avg_loser_duration_minutes=20.0,
        )

        assert analysis.best_hour == 10
        assert analysis.worst_hour == 14
        assert len(analysis.best_symbols) == 2
