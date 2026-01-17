# tests/journal/test_trade_logger.py
"""Tests for TradeLogger."""
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from src.execution.models import TrackedPosition
from src.journal.models import JournalEntry
from src.journal.settings import JournalSettings
from src.journal.trade_logger import TradeLogger
from src.scoring.models import Direction, ScoreComponents, ScoreTier, TradeRecommendation


def make_tracked_position(
    symbol: str = "NVDA",
    quantity: int = 100,
    entry_price: float = 140.0,
    stop_loss: float = 138.0,
    direction: Direction = Direction.LONG,
) -> TrackedPosition:
    """Create a TrackedPosition for testing."""
    return TrackedPosition(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        entry_time=datetime(2026, 1, 17, 9, 30, 0),
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


class TestTradeLogger:
    """Tests for TradeLogger."""

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
    def logger(self, settings: JournalSettings) -> TradeLogger:
        """Create a TradeLogger instance."""
        return TradeLogger(settings)

    async def test_log_entry_creates_file(
        self, logger: TradeLogger, temp_data_dir: Path
    ) -> None:
        """log_entry should create a new JSON file for the day."""
        position = make_tracked_position()
        recommendation = make_recommendation()

        trade_id = await logger.log_entry(position, recommendation)

        assert trade_id == "2026-01-17-NVDA-001"

        file_path = temp_data_dir / "2026-01-17.json"
        assert file_path.exists()

        with open(file_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["trade_id"] == "2026-01-17-NVDA-001"
        assert data[0]["symbol"] == "NVDA"
        assert data[0]["direction"] == "long"
        assert data[0]["entry_price"] == 140.0

    async def test_log_entry_appends_to_existing(
        self, logger: TradeLogger, temp_data_dir: Path
    ) -> None:
        """log_entry should append to existing file for same day."""
        position1 = make_tracked_position(symbol="NVDA")
        recommendation1 = make_recommendation(symbol="NVDA")

        position2 = make_tracked_position(symbol="AAPL", entry_price=180.0, stop_loss=178.0)
        recommendation2 = make_recommendation(symbol="AAPL", score=75.0)

        trade_id1 = await logger.log_entry(position1, recommendation1)
        trade_id2 = await logger.log_entry(position2, recommendation2)

        assert trade_id1 == "2026-01-17-NVDA-001"
        assert trade_id2 == "2026-01-17-AAPL-002"

        file_path = temp_data_dir / "2026-01-17.json"
        with open(file_path) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["symbol"] == "NVDA"
        assert data[1]["symbol"] == "AAPL"

    async def test_log_exit_updates_entry(
        self, logger: TradeLogger, temp_data_dir: Path
    ) -> None:
        """log_exit should update entry with exit details and calculate PnL."""
        position = make_tracked_position(
            entry_price=140.0, quantity=100, stop_loss=138.0
        )
        recommendation = make_recommendation()

        trade_id = await logger.log_entry(position, recommendation)

        await logger.log_exit(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 11, 30, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="take_profit",
        )

        file_path = temp_data_dir / "2026-01-17.json"
        with open(file_path) as f:
            data = json.load(f)

        entry = data[0]
        assert entry["exit_price"] == 144.0
        assert entry["exit_quantity"] == 100
        assert entry["exit_reason"] == "take_profit"

        # PnL for LONG: (144 - 140) * 100 = 400
        assert entry["pnl_dollars"] == 400.0

        # Percent: 400 / (140 * 100) * 100 = 2.857...
        assert abs(entry["pnl_percent"] - 2.857) < 0.01

        # R-multiple: gain / risk_per_share
        # risk_per_share = 140 - 138 = 2
        # gain_per_share = 144 - 140 = 4
        # R = 4 / 2 = 2.0
        assert entry["r_multiple"] == 2.0

    async def test_add_emotion_tag(
        self, logger: TradeLogger, temp_data_dir: Path
    ) -> None:
        """add_emotion_tag should update entry with emotion tag."""
        position = make_tracked_position()
        recommendation = make_recommendation()

        trade_id = await logger.log_entry(position, recommendation)
        await logger.add_emotion_tag(trade_id, "confident")

        file_path = temp_data_dir / "2026-01-17.json"
        with open(file_path) as f:
            data = json.load(f)

        assert data[0]["emotion_tag"] == "confident"

    async def test_add_notes(
        self, logger: TradeLogger, temp_data_dir: Path
    ) -> None:
        """add_notes should update entry with notes."""
        position = make_tracked_position()
        recommendation = make_recommendation()

        trade_id = await logger.log_entry(position, recommendation)
        await logger.add_notes(trade_id, "Followed the plan perfectly")

        file_path = temp_data_dir / "2026-01-17.json"
        with open(file_path) as f:
            data = json.load(f)

        assert data[0]["notes"] == "Followed the plan perfectly"

    async def test_get_entries_for_date(
        self, logger: TradeLogger, temp_data_dir: Path
    ) -> None:
        """get_entries_for_date should return all entries for a specific date."""
        position1 = make_tracked_position(symbol="NVDA")
        recommendation1 = make_recommendation(symbol="NVDA")

        position2 = make_tracked_position(symbol="AAPL", entry_price=180.0, stop_loss=178.0)
        recommendation2 = make_recommendation(symbol="AAPL")

        await logger.log_entry(position1, recommendation1)
        await logger.log_entry(position2, recommendation2)

        entries = await logger.get_entries_for_date(date(2026, 1, 17))

        assert len(entries) == 2
        assert all(isinstance(e, JournalEntry) for e in entries)
        assert entries[0].symbol == "NVDA"
        assert entries[1].symbol == "AAPL"

    async def test_get_entries_for_period(
        self, logger: TradeLogger, temp_data_dir: Path
    ) -> None:
        """get_entries_for_period should return entries within date range."""
        # Create entries for different dates by manually writing files
        jan_16_data = [
            {
                "trade_id": "2026-01-16-TSLA-001",
                "symbol": "TSLA",
                "direction": "long",
                "entry_time": "2026-01-16T10:00:00",
                "entry_price": 250.0,
                "entry_quantity": 50,
                "entry_reason": "test",
                "entry_score": 80.0,
                "stop_loss": 248.0,
                "exit_time": None,
                "exit_price": None,
                "exit_quantity": 0,
                "exit_reason": None,
                "pnl_dollars": 0.0,
                "pnl_percent": 0.0,
                "r_multiple": 0.0,
                "market_conditions": "test",
                "score_components": {
                    "sentiment_score": 80.0,
                    "technical_score": 80.0,
                    "sentiment_weight": 0.5,
                    "technical_weight": 0.5,
                    "confluence_bonus": 0.1,
                    "credibility_multiplier": 1.0,
                    "time_factor": 1.0,
                },
                "emotion_tag": None,
                "notes": None,
            }
        ]

        jan_17_data = [
            {
                "trade_id": "2026-01-17-NVDA-001",
                "symbol": "NVDA",
                "direction": "long",
                "entry_time": "2026-01-17T09:30:00",
                "entry_price": 140.0,
                "entry_quantity": 100,
                "entry_reason": "test",
                "entry_score": 85.0,
                "stop_loss": 138.0,
                "exit_time": None,
                "exit_price": None,
                "exit_quantity": 0,
                "exit_reason": None,
                "pnl_dollars": 0.0,
                "pnl_percent": 0.0,
                "r_multiple": 0.0,
                "market_conditions": "test",
                "score_components": {
                    "sentiment_score": 80.0,
                    "technical_score": 90.0,
                    "sentiment_weight": 0.5,
                    "technical_weight": 0.5,
                    "confluence_bonus": 0.1,
                    "credibility_multiplier": 1.0,
                    "time_factor": 1.0,
                },
                "emotion_tag": None,
                "notes": None,
            }
        ]

        jan_18_data = [
            {
                "trade_id": "2026-01-18-AAPL-001",
                "symbol": "AAPL",
                "direction": "short",
                "entry_time": "2026-01-18T11:00:00",
                "entry_price": 180.0,
                "entry_quantity": 75,
                "entry_reason": "test",
                "entry_score": 70.0,
                "stop_loss": 182.0,
                "exit_time": None,
                "exit_price": None,
                "exit_quantity": 0,
                "exit_reason": None,
                "pnl_dollars": 0.0,
                "pnl_percent": 0.0,
                "r_multiple": 0.0,
                "market_conditions": "test",
                "score_components": {
                    "sentiment_score": 70.0,
                    "technical_score": 70.0,
                    "sentiment_weight": 0.5,
                    "technical_weight": 0.5,
                    "confluence_bonus": 0.0,
                    "credibility_multiplier": 1.0,
                    "time_factor": 1.0,
                },
                "emotion_tag": None,
                "notes": None,
            }
        ]

        with open(temp_data_dir / "2026-01-16.json", "w") as f:
            json.dump(jan_16_data, f)
        with open(temp_data_dir / "2026-01-17.json", "w") as f:
            json.dump(jan_17_data, f)
        with open(temp_data_dir / "2026-01-18.json", "w") as f:
            json.dump(jan_18_data, f)

        entries = await logger.get_entries_for_period(
            start_date=date(2026, 1, 16),
            end_date=date(2026, 1, 17),
        )

        assert len(entries) == 2
        symbols = [e.symbol for e in entries]
        assert "TSLA" in symbols
        assert "NVDA" in symbols
        assert "AAPL" not in symbols
