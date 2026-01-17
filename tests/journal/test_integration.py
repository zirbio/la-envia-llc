# tests/journal/test_integration.py
"""Integration tests for the Journal module."""
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from src.execution.models import TrackedPosition
from src.journal import JournalManager, JournalSettings
from src.scoring.models import Direction, ScoreComponents, ScoreTier, TradeRecommendation


def make_tracked_position(
    symbol: str = "NVDA",
    quantity: int = 100,
    entry_price: float = 140.0,
    stop_loss: float = 138.0,
    direction: Direction = Direction.LONG,
    entry_time: datetime | None = None,
) -> TrackedPosition:
    """Create a TrackedPosition for testing."""
    return TrackedPosition(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        entry_time=entry_time or datetime(2026, 1, 17, 9, 30, 0),
        stop_loss=stop_loss,
        take_profit=entry_price + (entry_price - stop_loss) * 2.5,
        order_id=f"order-{symbol.lower()}",
        direction=direction,
    )


def make_recommendation(
    symbol: str = "NVDA",
    direction: Direction = Direction.LONG,
    score: float = 85.0,
    entry_price: float = 140.0,
    stop_loss: float = 138.0,
    components: ScoreComponents | None = None,
    timestamp: datetime | None = None,
) -> TradeRecommendation:
    """Create a TradeRecommendation for testing."""
    return TradeRecommendation(
        symbol=symbol,
        direction=direction,
        score=score,
        tier=ScoreTier.from_score(score),
        position_size_percent=5.0,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=entry_price + (entry_price - stop_loss) * 2.5,
        risk_reward_ratio=2.5,
        components=components
        or ScoreComponents(
            sentiment_score=80.0,
            technical_score=90.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        ),
        reasoning="Strong technical setup with positive sentiment",
        timestamp=timestamp or datetime(2026, 1, 17, 9, 29, 0),
    )


class TestJournalIntegration:
    """Integration tests for the Journal module."""

    @pytest.fixture
    def settings(self, tmp_path: Path) -> JournalSettings:
        """Create JournalSettings with temp data_dir."""
        data_dir = tmp_path / "trades"
        data_dir.mkdir()
        return JournalSettings(data_dir=str(data_dir))

    @pytest.fixture
    def sample_components(self) -> ScoreComponents:
        """Create a sample ScoreComponents instance."""
        return ScoreComponents(
            sentiment_score=80.0,
            technical_score=90.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )

    async def test_full_trade_lifecycle(
        self, settings: JournalSettings, sample_components: ScoreComponents
    ) -> None:
        """Test complete flow: open trade -> close trade -> get daily summary -> verify metrics and patterns."""
        manager = JournalManager(settings)
        trade_date = date(2026, 1, 17)

        # Open a trade
        position = make_tracked_position(
            symbol="NVDA",
            quantity=100,
            entry_price=140.0,
            stop_loss=138.0,
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 17, 9, 30, 0),
        )
        recommendation = make_recommendation(
            symbol="NVDA",
            direction=Direction.LONG,
            score=85.0,
            entry_price=140.0,
            stop_loss=138.0,
            components=sample_components,
            timestamp=datetime(2026, 1, 17, 9, 29, 0),
        )

        trade_id = await manager.on_trade_opened(position, recommendation)

        assert trade_id is not None
        assert trade_id == "2026-01-17-NVDA-001"

        # Verify open trade is recorded
        entries = await manager.get_entries_for_date(trade_date)
        assert len(entries) == 1
        assert entries[0].is_open
        assert entries[0].symbol == "NVDA"
        assert entries[0].direction == Direction.LONG
        assert entries[0].entry_price == 140.0
        assert entries[0].entry_quantity == 100
        assert entries[0].score_components == sample_components

        # Close the trade with profit
        await manager.on_trade_closed(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 11, 30, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="take_profit",
        )

        # Verify closed trade
        entries = await manager.get_entries_for_date(trade_date)
        assert len(entries) == 1
        closed_entry = entries[0]
        assert not closed_entry.is_open
        assert closed_entry.exit_price == 144.0
        assert closed_entry.exit_reason == "take_profit"
        # PnL: (144 - 140) * 100 = $400
        assert closed_entry.pnl_dollars == 400.0
        # R-multiple: gain_per_share / risk_per_share = 4 / 2 = 2.0
        assert closed_entry.r_multiple == 2.0

        # Get daily summary
        summary = await manager.get_daily_summary(trade_date)

        assert summary["date"] == trade_date
        assert summary["total_trades"] == 1
        assert summary["winning_trades"] == 1
        assert summary["losing_trades"] == 0
        assert summary["total_pnl"] == 400.0
        assert summary["win_rate"] == 1.0

        # Get weekly report and verify metrics and patterns
        report = await manager.get_weekly_report(trade_date)

        assert "metrics" in report
        assert "patterns" in report

        metrics = report["metrics"]
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 1.0
        assert metrics.total_pnl_dollars == 400.0
        assert metrics.avg_win_dollars == 400.0
        assert metrics.avg_win_r == 2.0
        assert metrics.best_trade is not None
        assert metrics.best_trade.symbol == "NVDA"

        patterns = report["patterns"]
        assert patterns.best_hour == 9  # Entry was at 9:30
        assert "NVDA" in [s[0] for s in patterns.best_symbols]

    async def test_multiple_trades_same_day(
        self, settings: JournalSettings, sample_components: ScoreComponents
    ) -> None:
        """Test two trades (one winner, one loser) on same day and verify combined metrics."""
        manager = JournalManager(settings)
        trade_date = date(2026, 1, 17)

        # Trade 1: Winning NVDA trade
        position1 = make_tracked_position(
            symbol="NVDA",
            quantity=100,
            entry_price=140.0,
            stop_loss=138.0,
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 17, 9, 30, 0),
        )
        recommendation1 = make_recommendation(
            symbol="NVDA",
            direction=Direction.LONG,
            score=85.0,
            entry_price=140.0,
            stop_loss=138.0,
            components=sample_components,
            timestamp=datetime(2026, 1, 17, 9, 29, 0),
        )

        trade_id1 = await manager.on_trade_opened(position1, recommendation1)
        assert trade_id1 is not None

        await manager.on_trade_closed(
            trade_id=trade_id1,
            exit_time=datetime(2026, 1, 17, 10, 30, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="take_profit",
        )

        # Trade 2: Losing AAPL trade
        position2 = make_tracked_position(
            symbol="AAPL",
            quantity=50,
            entry_price=180.0,
            stop_loss=176.0,
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 17, 11, 0, 0),
        )
        losing_components = ScoreComponents(
            sentiment_score=65.0,
            technical_score=70.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.05,
            credibility_multiplier=0.95,
            time_factor=0.9,
        )
        recommendation2 = make_recommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=68.0,
            entry_price=180.0,
            stop_loss=176.0,
            components=losing_components,
            timestamp=datetime(2026, 1, 17, 10, 59, 0),
        )

        trade_id2 = await manager.on_trade_opened(position2, recommendation2)
        assert trade_id2 is not None

        await manager.on_trade_closed(
            trade_id=trade_id2,
            exit_time=datetime(2026, 1, 17, 12, 0, 0),
            exit_price=176.0,
            exit_quantity=50,
            exit_reason="stop_loss",
        )

        # Verify both trades are recorded
        entries = await manager.get_entries_for_date(trade_date)
        assert len(entries) == 2

        # Verify trade details
        nvda_entry = next(e for e in entries if e.symbol == "NVDA")
        aapl_entry = next(e for e in entries if e.symbol == "AAPL")

        # NVDA: (144 - 140) * 100 = $400 profit
        assert nvda_entry.pnl_dollars == 400.0
        # AAPL: (176 - 180) * 50 = -$200 loss
        assert aapl_entry.pnl_dollars == -200.0

        # Verify daily summary with combined metrics
        summary = await manager.get_daily_summary(trade_date)

        assert summary["total_trades"] == 2
        assert summary["winning_trades"] == 1
        assert summary["losing_trades"] == 1
        assert summary["total_pnl"] == 200.0  # 400 - 200
        assert summary["win_rate"] == 0.5

        # Verify weekly report metrics
        report = await manager.get_weekly_report(trade_date)
        metrics = report["metrics"]

        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5
        assert metrics.total_pnl_dollars == 200.0
        assert metrics.avg_win_dollars == 400.0
        # avg_loss_dollars is stored as absolute value (positive)
        assert metrics.avg_loss_dollars == 200.0
        assert metrics.best_trade is not None
        assert metrics.best_trade.symbol == "NVDA"
        assert metrics.worst_trade is not None
        assert metrics.worst_trade.symbol == "AAPL"

        # Verify profit factor: gross_profit / abs(gross_loss) = 400 / 200 = 2.0
        assert metrics.profit_factor == 2.0

    async def test_emotion_tags_and_notes(
        self, settings: JournalSettings, sample_components: ScoreComponents
    ) -> None:
        """Test adding emotion tags and notes to trades."""
        manager = JournalManager(settings)
        trade_date = date(2026, 1, 17)

        # Open a trade
        position = make_tracked_position(
            symbol="TSLA",
            quantity=25,
            entry_price=250.0,
            stop_loss=245.0,
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 17, 10, 0, 0),
        )
        recommendation = make_recommendation(
            symbol="TSLA",
            direction=Direction.LONG,
            score=78.0,
            entry_price=250.0,
            stop_loss=245.0,
            components=sample_components,
            timestamp=datetime(2026, 1, 17, 9, 59, 0),
        )

        trade_id = await manager.on_trade_opened(position, recommendation)
        assert trade_id is not None

        # Add emotion tag
        await manager.add_emotion_tag(trade_id, "confident")

        # Verify emotion tag is recorded
        entries = await manager.get_entries_for_date(trade_date)
        assert len(entries) == 1
        assert entries[0].emotion_tag == "confident"
        assert entries[0].notes is None

        # Add notes
        await manager.add_notes(trade_id, "Strong breakout setup. Followed the plan.")

        # Verify notes are recorded
        entries = await manager.get_entries_for_date(trade_date)
        assert len(entries) == 1
        assert entries[0].emotion_tag == "confident"
        assert entries[0].notes == "Strong breakout setup. Followed the plan."

        # Close the trade
        await manager.on_trade_closed(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 14, 0, 0),
            exit_price=260.0,
            exit_quantity=25,
            exit_reason="take_profit",
        )

        # Verify emotion tag and notes persist after closing
        entries = await manager.get_entries_for_date(trade_date)
        assert len(entries) == 1
        closed_entry = entries[0]
        assert not closed_entry.is_open
        assert closed_entry.emotion_tag == "confident"
        assert closed_entry.notes == "Strong breakout setup. Followed the plan."
        # PnL: (260 - 250) * 25 = $250
        assert closed_entry.pnl_dollars == 250.0

        # Update notes after trade is closed
        await manager.add_notes(
            trade_id, "Strong breakout setup. Followed the plan. Great execution!"
        )

        entries = await manager.get_entries_for_date(trade_date)
        assert (
            entries[0].notes
            == "Strong breakout setup. Followed the plan. Great execution!"
        )

        # Add different emotion tag to reflect post-trade feelings
        await manager.add_emotion_tag(trade_id, "satisfied")

        entries = await manager.get_entries_for_date(trade_date)
        assert entries[0].emotion_tag == "satisfied"
