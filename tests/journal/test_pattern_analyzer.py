# tests/journal/test_pattern_analyzer.py
"""Tests for PatternAnalyzer."""
from datetime import datetime

from src.journal.models import JournalEntry, PatternAnalysis
from src.journal.pattern_analyzer import PatternAnalyzer
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
    entry_time: datetime,
    exit_time: datetime,
    entry_reason: str = "test_setup",
) -> JournalEntry:
    """Create a closed journal entry for testing."""
    return JournalEntry(
        trade_id=trade_id,
        symbol=symbol,
        direction=Direction.LONG,
        entry_time=entry_time,
        entry_price=100.0,
        entry_quantity=100,
        entry_reason=entry_reason,
        entry_score=85.0,
        stop_loss=98.0,
        exit_time=exit_time,
        exit_price=100.0 + (pnl_dollars / 100),
        exit_quantity=100,
        exit_reason="take_profit",
        pnl_dollars=pnl_dollars,
        pnl_percent=pnl_dollars / 100.0,
        r_multiple=pnl_dollars / 200.0,
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


class TestPatternAnalyzer:
    """Tests for PatternAnalyzer."""

    def test_analyze_empty_entries(self) -> None:
        """analyze should return defaults when no entries provided."""
        analyzer = PatternAnalyzer()

        result = analyzer.analyze(entries=[])

        assert result.best_hour == -1
        assert result.worst_hour == -1
        assert result.best_day_of_week == -1
        assert result.best_symbols == []
        assert result.worst_symbols == []
        assert result.best_setups == []
        assert result.worst_setups == []
        assert result.avg_winner_duration_minutes == 0.0
        assert result.avg_loser_duration_minutes == 0.0

    def test_analyze_best_worst_hour(self) -> None:
        """analyze should find best and worst trading hours by avg PnL."""
        analyzer = PatternAnalyzer()
        entries = [
            # Hour 9: avg PnL = (100 + 50) / 2 = 75
            make_closed_entry(
                "1", "NVDA", 100.0,
                datetime(2026, 1, 17, 9, 30, 0),
                datetime(2026, 1, 17, 9, 45, 0),
            ),
            make_closed_entry(
                "2", "AAPL", 50.0,
                datetime(2026, 1, 17, 9, 45, 0),
                datetime(2026, 1, 17, 10, 0, 0),
            ),
            # Hour 10: avg PnL = -100
            make_closed_entry(
                "3", "TSLA", -100.0,
                datetime(2026, 1, 17, 10, 15, 0),
                datetime(2026, 1, 17, 10, 30, 0),
            ),
            # Hour 14: avg PnL = 200
            make_closed_entry(
                "4", "MSFT", 200.0,
                datetime(2026, 1, 17, 14, 0, 0),
                datetime(2026, 1, 17, 14, 30, 0),
            ),
        ]

        result = analyzer.analyze(entries=entries)

        assert result.best_hour == 14
        assert result.worst_hour == 10

    def test_analyze_best_day_of_week(self) -> None:
        """analyze should find best day of week by avg PnL."""
        analyzer = PatternAnalyzer()
        # Monday (0): avg = (100 - 20) / 2 = 40
        # Wednesday (2): avg = 200
        # Friday (4): avg = -50
        entries = [
            make_closed_entry(
                "1", "NVDA", 100.0,
                datetime(2026, 1, 12, 9, 30, 0),  # Monday
                datetime(2026, 1, 12, 10, 0, 0),
            ),
            make_closed_entry(
                "2", "AAPL", -20.0,
                datetime(2026, 1, 12, 10, 30, 0),  # Monday
                datetime(2026, 1, 12, 11, 0, 0),
            ),
            make_closed_entry(
                "3", "TSLA", 200.0,
                datetime(2026, 1, 14, 9, 30, 0),  # Wednesday
                datetime(2026, 1, 14, 10, 0, 0),
            ),
            make_closed_entry(
                "4", "MSFT", -50.0,
                datetime(2026, 1, 16, 9, 30, 0),  # Friday
                datetime(2026, 1, 16, 10, 0, 0),
            ),
        ]

        result = analyzer.analyze(entries=entries)

        assert result.best_day_of_week == 2  # Wednesday

    def test_analyze_best_worst_symbols(self) -> None:
        """analyze should rank best and worst symbols by avg PnL."""
        analyzer = PatternAnalyzer()
        entries = [
            # NVDA: avg = (100 + 150) / 2 = 125
            make_closed_entry(
                "1", "NVDA", 100.0,
                datetime(2026, 1, 17, 9, 30, 0),
                datetime(2026, 1, 17, 10, 0, 0),
            ),
            make_closed_entry(
                "2", "NVDA", 150.0,
                datetime(2026, 1, 17, 10, 30, 0),
                datetime(2026, 1, 17, 11, 0, 0),
            ),
            # AAPL: avg = 200
            make_closed_entry(
                "3", "AAPL", 200.0,
                datetime(2026, 1, 17, 11, 30, 0),
                datetime(2026, 1, 17, 12, 0, 0),
            ),
            # TSLA: avg = -75
            make_closed_entry(
                "4", "TSLA", -75.0,
                datetime(2026, 1, 17, 12, 30, 0),
                datetime(2026, 1, 17, 13, 0, 0),
            ),
            # MSFT: avg = -150
            make_closed_entry(
                "5", "MSFT", -150.0,
                datetime(2026, 1, 17, 13, 30, 0),
                datetime(2026, 1, 17, 14, 0, 0),
            ),
        ]

        result = analyzer.analyze(entries=entries)

        # Best symbols: sorted by avg PnL descending
        assert len(result.best_symbols) <= 5
        assert result.best_symbols[0] == ("AAPL", 200.0)
        assert result.best_symbols[1] == ("NVDA", 125.0)

        # Worst symbols: sorted by avg PnL ascending
        assert len(result.worst_symbols) <= 5
        assert result.worst_symbols[0] == ("MSFT", -150.0)
        assert result.worst_symbols[1] == ("TSLA", -75.0)

    def test_analyze_best_worst_setups(self) -> None:
        """analyze should rank best and worst setups by avg PnL."""
        analyzer = PatternAnalyzer()
        entries = [
            # breakout: avg = (100 + 80) / 2 = 90
            make_closed_entry(
                "1", "NVDA", 100.0,
                datetime(2026, 1, 17, 9, 30, 0),
                datetime(2026, 1, 17, 10, 0, 0),
                entry_reason="breakout",
            ),
            make_closed_entry(
                "2", "AAPL", 80.0,
                datetime(2026, 1, 17, 10, 30, 0),
                datetime(2026, 1, 17, 11, 0, 0),
                entry_reason="breakout",
            ),
            # reversal: avg = -50
            make_closed_entry(
                "3", "TSLA", -50.0,
                datetime(2026, 1, 17, 11, 30, 0),
                datetime(2026, 1, 17, 12, 0, 0),
                entry_reason="reversal",
            ),
            # momentum: avg = 150
            make_closed_entry(
                "4", "MSFT", 150.0,
                datetime(2026, 1, 17, 12, 30, 0),
                datetime(2026, 1, 17, 13, 0, 0),
                entry_reason="momentum",
            ),
        ]

        result = analyzer.analyze(entries=entries)

        # Best setups: sorted by avg PnL descending
        assert len(result.best_setups) <= 5
        assert result.best_setups[0] == ("momentum", 150.0)
        assert result.best_setups[1] == ("breakout", 90.0)

        # Worst setups: sorted by avg PnL ascending
        assert len(result.worst_setups) <= 5
        assert result.worst_setups[0] == ("reversal", -50.0)

    def test_analyze_duration(self) -> None:
        """analyze should calculate avg winner and loser durations."""
        analyzer = PatternAnalyzer()
        entries = [
            # Winner: 30 minutes
            make_closed_entry(
                "1", "NVDA", 100.0,
                datetime(2026, 1, 17, 9, 30, 0),
                datetime(2026, 1, 17, 10, 0, 0),
            ),
            # Winner: 60 minutes
            make_closed_entry(
                "2", "AAPL", 50.0,
                datetime(2026, 1, 17, 10, 0, 0),
                datetime(2026, 1, 17, 11, 0, 0),
            ),
            # Loser: 15 minutes
            make_closed_entry(
                "3", "TSLA", -50.0,
                datetime(2026, 1, 17, 11, 0, 0),
                datetime(2026, 1, 17, 11, 15, 0),
            ),
            # Loser: 45 minutes
            make_closed_entry(
                "4", "MSFT", -100.0,
                datetime(2026, 1, 17, 12, 0, 0),
                datetime(2026, 1, 17, 12, 45, 0),
            ),
        ]

        result = analyzer.analyze(entries=entries)

        # Avg winner duration: (30 + 60) / 2 = 45 minutes
        assert result.avg_winner_duration_minutes == 45.0
        # Avg loser duration: (15 + 45) / 2 = 30 minutes
        assert result.avg_loser_duration_minutes == 30.0
