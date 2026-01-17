# tests/journal/test_journal_manager.py
"""Tests for JournalManager."""
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from src.execution.models import TrackedPosition
from src.journal.journal_manager import JournalManager
from src.journal.settings import JournalSettings
from src.scoring.models import (
    Direction,
    ScoreComponents,
    ScoreTier,
    TradeRecommendation,
)


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
        take_profit=145.0,
        order_id="order-123",
        direction=direction,
    )


def make_recommendation(
    symbol: str = "NVDA",
    direction: Direction = Direction.LONG,
    score: float = 85.0,
) -> TradeRecommendation:
    """Create a TradeRecommendation for testing."""
    return TradeRecommendation(
        symbol=symbol,
        direction=direction,
        score=score,
        tier=ScoreTier.from_score(score),
        position_size_percent=5.0,
        entry_price=140.0,
        stop_loss=138.0,
        take_profit=145.0,
        risk_reward_ratio=2.5,
        components=ScoreComponents(
            sentiment_score=80.0,
            technical_score=90.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        ),
        reasoning="Strong technical setup with positive sentiment",
        timestamp=datetime(2026, 1, 17, 9, 29, 0),
    )


class TestJournalManager:
    """Tests for JournalManager."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path: Path) -> Path:
        """Create a temporary data directory."""
        data_dir = tmp_path / "trades"
        data_dir.mkdir()
        return data_dir

    @pytest.fixture
    def settings(self, temp_data_dir: Path) -> JournalSettings:
        """Create settings with temporary data directory."""
        return JournalSettings(data_dir=str(temp_data_dir))

    @pytest.fixture
    def disabled_settings(self, temp_data_dir: Path) -> JournalSettings:
        """Create disabled settings."""
        return JournalSettings(data_dir=str(temp_data_dir), enabled=False)

    @pytest.fixture
    def manager(self, settings: JournalSettings) -> JournalManager:
        """Create a JournalManager instance."""
        return JournalManager(settings)

    @pytest.fixture
    def disabled_manager(self, disabled_settings: JournalSettings) -> JournalManager:
        """Create a disabled JournalManager instance."""
        return JournalManager(disabled_settings)

    async def test_on_trade_opened(self, manager: JournalManager) -> None:
        """on_trade_opened should log entry and return trade_id."""
        position = make_tracked_position()
        recommendation = make_recommendation()

        trade_id = await manager.on_trade_opened(position, recommendation)

        assert trade_id is not None
        assert trade_id == "2026-01-17-NVDA-001"

        entries = await manager.get_entries_for_date(date(2026, 1, 17))
        assert len(entries) == 1
        assert entries[0].symbol == "NVDA"

    async def test_on_trade_closed(self, manager: JournalManager) -> None:
        """on_trade_closed should log exit with PnL calculation."""
        position = make_tracked_position()
        recommendation = make_recommendation()

        trade_id = await manager.on_trade_opened(position, recommendation)
        assert trade_id is not None

        await manager.on_trade_closed(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 11, 30, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="take_profit",
        )

        entries = await manager.get_entries_for_date(date(2026, 1, 17))
        assert len(entries) == 1
        entry = entries[0]
        assert entry.exit_price == 144.0
        assert entry.pnl_dollars == 400.0  # (144 - 140) * 100

    async def test_get_daily_summary(self, manager: JournalManager) -> None:
        """get_daily_summary should return dict with trade statistics."""
        position1 = make_tracked_position(symbol="NVDA")
        recommendation1 = make_recommendation(symbol="NVDA")
        trade_id1 = await manager.on_trade_opened(position1, recommendation1)
        assert trade_id1 is not None
        await manager.on_trade_closed(
            trade_id=trade_id1,
            exit_time=datetime(2026, 1, 17, 10, 0, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="take_profit",
        )

        position2 = make_tracked_position(
            symbol="AAPL", entry_price=180.0, stop_loss=178.0
        )
        recommendation2 = make_recommendation(symbol="AAPL")
        trade_id2 = await manager.on_trade_opened(position2, recommendation2)
        assert trade_id2 is not None
        await manager.on_trade_closed(
            trade_id=trade_id2,
            exit_time=datetime(2026, 1, 17, 11, 0, 0),
            exit_price=177.0,
            exit_quantity=100,
            exit_reason="stop_loss",
        )

        summary = await manager.get_daily_summary(date(2026, 1, 17))

        assert summary["date"] == date(2026, 1, 17)
        assert summary["total_trades"] == 2
        assert summary["winning_trades"] == 1
        assert summary["losing_trades"] == 1
        assert summary["total_pnl"] == 100.0  # 400 - 300
        assert summary["win_rate"] == 0.5

    async def test_get_weekly_report(self, manager: JournalManager) -> None:
        """get_weekly_report should return dict with metrics and patterns."""
        position = make_tracked_position()
        recommendation = make_recommendation()
        trade_id = await manager.on_trade_opened(position, recommendation)
        assert trade_id is not None
        await manager.on_trade_closed(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 10, 0, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="take_profit",
        )

        end_date = date(2026, 1, 17)
        report = await manager.get_weekly_report(end_date)

        assert report["start_date"] == end_date - timedelta(days=6)
        assert report["end_date"] == end_date
        assert "metrics" in report
        assert "patterns" in report
        assert report["metrics"].total_trades == 1
        assert report["metrics"].winning_trades == 1

    async def test_disabled_does_nothing(
        self, disabled_manager: JournalManager
    ) -> None:
        """When disabled, operations should return None/empty and not log."""
        position = make_tracked_position()
        recommendation = make_recommendation()

        trade_id = await disabled_manager.on_trade_opened(position, recommendation)
        assert trade_id is None

        await disabled_manager.on_trade_closed(
            trade_id="fake-id",
            exit_time=datetime(2026, 1, 17, 10, 0, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="test",
        )

        await disabled_manager.add_emotion_tag("fake-id", "confident")
        await disabled_manager.add_notes("fake-id", "test notes")

        entries = await disabled_manager.get_entries_for_date(date(2026, 1, 17))
        assert entries == []

        summary = await disabled_manager.get_daily_summary(date(2026, 1, 17))
        assert summary["total_trades"] == 0
