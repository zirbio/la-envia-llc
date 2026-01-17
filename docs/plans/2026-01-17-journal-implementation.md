# Journal Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a trading journal that auto-logs trades, calculates performance metrics, and analyzes trading patterns.

**Architecture:** Four-component design with TradeLogger (persistence), MetricsCalculator (statistics), PatternAnalyzer (insights), and JournalManager (orchestration). JSON file storage per day in `data/trades/`.

**Tech Stack:** Python 3.13, Pydantic for settings, dataclasses for models, pytest-asyncio for testing, aiofiles for async file I/O.

---

## Task 1: Create Journal Models

**Files:**
- Create: `src/journal/__init__.py`
- Create: `src/journal/models.py`
- Create: `tests/journal/__init__.py`
- Create: `tests/journal/test_models.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/journal tests/journal
touch src/journal/__init__.py tests/journal/__init__.py
```

**Step 2: Write the failing test**

```python
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
```

**Step 3: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.journal'"

**Step 4: Write minimal implementation**

```python
# src/journal/models.py
"""Data models for the trading journal."""
from dataclasses import dataclass
from datetime import datetime

from src.scoring.models import Direction, ScoreComponents


@dataclass
class JournalEntry:
    """A single trade entry in the journal.

    Attributes:
        trade_id: Unique identifier (format: YYYY-MM-DD-SYMBOL-NNN).
        symbol: Stock ticker symbol.
        direction: Trade direction (LONG or SHORT).
        entry_time: When the trade was opened.
        entry_price: Entry price per share.
        entry_quantity: Number of shares entered.
        entry_reason: Reason for entering (from signal).
        entry_score: Original signal score.
        stop_loss: Initial stop loss price.
        exit_time: When the trade was closed (None if open).
        exit_price: Exit price per share (None if open).
        exit_quantity: Number of shares exited.
        exit_reason: Reason for exiting.
        pnl_dollars: Profit/loss in dollars.
        pnl_percent: Profit/loss as percentage.
        r_multiple: PnL relative to initial risk.
        market_conditions: Market state at entry.
        score_components: Breakdown of entry score.
        emotion_tag: Optional manual emotion tag.
        notes: Optional manual notes.
    """

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
    """Calculated trading performance metrics.

    Attributes:
        period_days: Number of days in the calculation period.
        total_trades: Total number of trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        win_rate: Ratio of winning trades (0-1).
        profit_factor: Gross profit / gross loss.
        expectancy: Average PnL per trade in R.
        avg_win_dollars: Average winning trade in dollars.
        avg_loss_dollars: Average losing trade in dollars.
        avg_win_r: Average R-multiple for winners.
        avg_loss_r: Average R-multiple for losers.
        total_pnl_dollars: Total profit/loss in dollars.
        total_pnl_percent: Total profit/loss as percentage.
        max_drawdown_percent: Maximum drawdown percentage.
        sharpe_ratio: Annualized Sharpe ratio.
        best_trade: Best performing trade.
        worst_trade: Worst performing trade.
    """

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
    """Analysis of trading patterns.

    Attributes:
        best_hour: Hour with highest average PnL (0-23).
        worst_hour: Hour with lowest average PnL.
        best_day_of_week: Day with highest average PnL (0=Monday).
        best_symbols: Symbols with highest average PnL.
        worst_symbols: Symbols with lowest average PnL.
        best_setups: Entry reasons with highest average PnL.
        worst_setups: Entry reasons with lowest average PnL.
        avg_winner_duration_minutes: Average hold time for winners.
        avg_loser_duration_minutes: Average hold time for losers.
    """

    best_hour: int
    worst_hour: int
    best_day_of_week: int

    best_symbols: list[tuple[str, float]]
    worst_symbols: list[tuple[str, float]]

    best_setups: list[tuple[str, float]]
    worst_setups: list[tuple[str, float]]

    avg_winner_duration_minutes: float
    avg_loser_duration_minutes: float
```

```python
# src/journal/__init__.py
"""Journal module for trading history and metrics."""

from .models import JournalEntry, PatternAnalysis, TradingMetrics

__all__ = [
    "JournalEntry",
    "PatternAnalysis",
    "TradingMetrics",
]
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_models.py -v`
Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add src/journal/ tests/journal/
git commit -m "feat(journal): add JournalEntry, TradingMetrics, PatternAnalysis models"
```

---

## Task 2: Create JournalSettings

**Files:**
- Create: `src/journal/settings.py`
- Create: `tests/journal/test_settings.py`
- Modify: `src/journal/__init__.py`

**Step 1: Write the failing test**

```python
# tests/journal/test_settings.py
"""Tests for journal settings."""
import pytest

from src.journal.settings import JournalSettings


class TestJournalSettings:
    """Tests for JournalSettings."""

    def test_default_settings(self):
        """Default settings should have sensible values."""
        settings = JournalSettings()

        assert settings.enabled is True
        assert settings.data_dir == "data/trades"
        assert settings.auto_log_entries is True
        assert settings.auto_log_exits is True
        assert settings.default_period_days == 30
        assert settings.weekly_report_enabled is True
        assert settings.weekly_report_day == "saturday"

    def test_custom_settings(self):
        """Settings should accept custom values."""
        settings = JournalSettings(
            enabled=False,
            data_dir="custom/path",
            default_period_days=60,
        )

        assert settings.enabled is False
        assert settings.data_dir == "custom/path"
        assert settings.default_period_days == 60

    def test_weekly_report_day_validation(self):
        """weekly_report_day should only accept valid days."""
        with pytest.raises(ValueError):
            JournalSettings(weekly_report_day="invalid")

    def test_valid_weekday_names(self):
        """All weekday names should be valid."""
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            settings = JournalSettings(weekly_report_day=day)
            assert settings.weekly_report_day == day
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_settings.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/journal/settings.py
"""Settings for the journal module."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


WeekDay = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class JournalSettings(BaseModel):
    """Configuration settings for the trading journal.

    Attributes:
        enabled: Whether journaling is enabled.
        data_dir: Directory to store trade JSON files.
        auto_log_entries: Auto-log when trades open.
        auto_log_exits: Auto-log when trades close.
        default_period_days: Default period for metrics calculation.
        weekly_report_enabled: Generate weekly reports.
        weekly_report_day: Day to generate weekly report.
    """

    enabled: bool = True
    data_dir: str = "data/trades"

    auto_log_entries: bool = True
    auto_log_exits: bool = True

    default_period_days: int = Field(default=30, ge=1, le=365)

    weekly_report_enabled: bool = True
    weekly_report_day: WeekDay = "saturday"

    @field_validator("weekly_report_day")
    @classmethod
    def validate_weekday(cls, v: str) -> str:
        """Validate that weekly_report_day is a valid weekday."""
        valid_days = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
        if v.lower() not in valid_days:
            raise ValueError(f"Invalid weekday: {v}. Must be one of {valid_days}")
        return v.lower()
```

Update `src/journal/__init__.py`:

```python
# src/journal/__init__.py
"""Journal module for trading history and metrics."""

from .models import JournalEntry, PatternAnalysis, TradingMetrics
from .settings import JournalSettings

__all__ = [
    "JournalEntry",
    "JournalSettings",
    "PatternAnalysis",
    "TradingMetrics",
]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_settings.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/journal/settings.py tests/journal/test_settings.py src/journal/__init__.py
git commit -m "feat(journal): add JournalSettings with Pydantic validation"
```

---

## Task 3: Create TradeLogger

**Files:**
- Create: `src/journal/trade_logger.py`
- Create: `tests/journal/test_trade_logger.py`
- Modify: `src/journal/__init__.py`

**Step 1: Write the failing test**

```python
# tests/journal/test_trade_logger.py
"""Tests for trade logger."""
import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.journal.trade_logger import TradeLogger
from src.journal.settings import JournalSettings
from src.journal.models import JournalEntry
from src.scoring.models import Direction, ScoreComponents, TradeRecommendation, ScoreTier
from src.execution.models import TrackedPosition


@pytest.fixture
def settings(tmp_path):
    """Create settings with temp directory."""
    return JournalSettings(data_dir=str(tmp_path / "trades"))


@pytest.fixture
def trade_logger(settings):
    """Create a TradeLogger instance."""
    return TradeLogger(settings)


@pytest.fixture
def sample_position():
    """Create a sample tracked position."""
    return TrackedPosition(
        symbol="NVDA",
        quantity=322,
        entry_price=140.76,
        entry_time=datetime(2026, 1, 17, 9, 33, 0),
        stop_loss=139.20,
        take_profit=145.40,
        order_id="order-123",
        direction=Direction.LONG,
    )


@pytest.fixture
def sample_recommendation():
    """Create a sample trade recommendation."""
    components = ScoreComponents(
        sentiment_score=80.0,
        technical_score=75.0,
        sentiment_weight=0.5,
        technical_weight=0.5,
        confluence_bonus=0.1,
        credibility_multiplier=1.0,
        time_factor=1.0,
    )
    return TradeRecommendation(
        symbol="NVDA",
        direction=Direction.LONG,
        score=88.0,
        tier=ScoreTier.STRONG,
        position_size_percent=90.0,
        entry_price=140.76,
        stop_loss=139.20,
        take_profit=145.40,
        risk_reward_ratio=3.0,
        components=components,
        reasoning="unusual_whales_sweep_alert",
        timestamp=datetime(2026, 1, 17, 9, 33, 0),
    )


class TestTradeLogger:
    """Tests for TradeLogger."""

    @pytest.mark.asyncio
    async def test_log_entry_creates_file(self, trade_logger, sample_position, sample_recommendation, tmp_path):
        """log_entry should create a JSON file for the day."""
        trade_id = await trade_logger.log_entry(sample_position, sample_recommendation)

        assert trade_id == "2026-01-17-NVDA-001"

        file_path = tmp_path / "trades" / "2026-01-17.json"
        assert file_path.exists()

        with open(file_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["trade_id"] == trade_id
        assert data[0]["symbol"] == "NVDA"

    @pytest.mark.asyncio
    async def test_log_entry_appends_to_existing(self, trade_logger, sample_position, sample_recommendation, tmp_path):
        """log_entry should append to existing daily file."""
        trade_id_1 = await trade_logger.log_entry(sample_position, sample_recommendation)

        # Create second position
        position_2 = TrackedPosition(
            symbol="AAPL",
            quantity=100,
            entry_price=180.50,
            entry_time=datetime(2026, 1, 17, 10, 15, 0),
            stop_loss=178.00,
            take_profit=185.00,
            order_id="order-456",
            direction=Direction.LONG,
        )
        recommendation_2 = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=75.0,
            tier=ScoreTier.MODERATE,
            position_size_percent=50.0,
            entry_price=180.50,
            stop_loss=178.00,
            take_profit=185.00,
            risk_reward_ratio=2.0,
            components=sample_recommendation.components,
            reasoning="momentum_breakout",
            timestamp=datetime(2026, 1, 17, 10, 15, 0),
        )

        trade_id_2 = await trade_logger.log_entry(position_2, recommendation_2)

        assert trade_id_2 == "2026-01-17-AAPL-002"

        file_path = tmp_path / "trades" / "2026-01-17.json"
        with open(file_path) as f:
            data = json.load(f)

        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_log_exit_updates_entry(self, trade_logger, sample_position, sample_recommendation, tmp_path):
        """log_exit should update the entry with exit data."""
        trade_id = await trade_logger.log_entry(sample_position, sample_recommendation)

        await trade_logger.log_exit(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 11, 45, 0),
            exit_price=142.38,
            exit_quantity=322,
            exit_reason="trailing_stop",
        )

        file_path = tmp_path / "trades" / "2026-01-17.json"
        with open(file_path) as f:
            data = json.load(f)

        entry = data[0]
        assert entry["exit_price"] == 142.38
        assert entry["exit_reason"] == "trailing_stop"
        assert entry["pnl_dollars"] == pytest.approx(521.44, rel=0.01)
        assert entry["pnl_percent"] == pytest.approx(1.15, rel=0.01)

    @pytest.mark.asyncio
    async def test_add_emotion_tag(self, trade_logger, sample_position, sample_recommendation, tmp_path):
        """add_emotion_tag should update the entry."""
        trade_id = await trade_logger.log_entry(sample_position, sample_recommendation)

        await trade_logger.add_emotion_tag(trade_id, "confident")

        file_path = tmp_path / "trades" / "2026-01-17.json"
        with open(file_path) as f:
            data = json.load(f)

        assert data[0]["emotion_tag"] == "confident"

    @pytest.mark.asyncio
    async def test_add_notes(self, trade_logger, sample_position, sample_recommendation, tmp_path):
        """add_notes should update the entry."""
        trade_id = await trade_logger.log_entry(sample_position, sample_recommendation)

        await trade_logger.add_notes(trade_id, "Great setup, followed plan")

        file_path = tmp_path / "trades" / "2026-01-17.json"
        with open(file_path) as f:
            data = json.load(f)

        assert data[0]["notes"] == "Great setup, followed plan"

    @pytest.mark.asyncio
    async def test_get_entries_for_date(self, trade_logger, sample_position, sample_recommendation, tmp_path):
        """get_entries_for_date should return all entries for a day."""
        await trade_logger.log_entry(sample_position, sample_recommendation)

        entries = await trade_logger.get_entries_for_date(datetime(2026, 1, 17))

        assert len(entries) == 1
        assert entries[0].symbol == "NVDA"

    @pytest.mark.asyncio
    async def test_get_entries_for_period(self, trade_logger, sample_position, sample_recommendation, tmp_path):
        """get_entries_for_period should return entries within date range."""
        await trade_logger.log_entry(sample_position, sample_recommendation)

        entries = await trade_logger.get_entries_for_period(
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 1, 31),
        )

        assert len(entries) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_trade_logger.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/journal/trade_logger.py
"""Trade logger for persisting trade entries."""
import json
import logging
from datetime import datetime
from pathlib import Path

import aiofiles

from src.execution.models import TrackedPosition
from src.scoring.models import TradeRecommendation

from .models import JournalEntry
from .settings import JournalSettings

logger = logging.getLogger(__name__)


class TradeLogger:
    """Logs trades to JSON files.

    One JSON file per day containing all trades for that day.
    """

    def __init__(self, settings: JournalSettings) -> None:
        """Initialize the trade logger.

        Args:
            settings: Journal configuration settings.
        """
        self._settings = settings
        self._data_dir = Path(settings.data_dir)
        self._trade_counts: dict[str, int] = {}  # date -> count for ID generation

    async def log_entry(
        self,
        position: TrackedPosition,
        recommendation: TradeRecommendation,
    ) -> str:
        """Log a new trade entry.

        Args:
            position: The tracked position that was opened.
            recommendation: The trade recommendation that triggered the trade.

        Returns:
            The generated trade ID.
        """
        date_str = position.entry_time.strftime("%Y-%m-%d")
        trade_id = self._generate_trade_id(date_str, position.symbol)

        entry = JournalEntry(
            trade_id=trade_id,
            symbol=position.symbol,
            direction=position.direction,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            entry_quantity=position.quantity,
            entry_reason=recommendation.reasoning,
            entry_score=recommendation.score,
            stop_loss=position.stop_loss,
            exit_time=None,
            exit_price=None,
            exit_quantity=0,
            exit_reason=None,
            pnl_dollars=0.0,
            pnl_percent=0.0,
            r_multiple=0.0,
            market_conditions=self._determine_market_conditions(recommendation),
            score_components=recommendation.components,
            emotion_tag=None,
            notes=None,
        )

        await self._save_entry(entry, date_str)
        logger.info(f"Logged entry for trade {trade_id}")

        return trade_id

    async def log_exit(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        exit_quantity: int,
        exit_reason: str,
    ) -> None:
        """Log a trade exit.

        Args:
            trade_id: The trade ID to update.
            exit_time: When the trade was closed.
            exit_price: The exit price per share.
            exit_quantity: Number of shares exited.
            exit_reason: Reason for the exit.
        """
        date_str = trade_id.split("-")[0] + "-" + trade_id.split("-")[1] + "-" + trade_id.split("-")[2]
        entries = await self._load_entries(date_str)

        for entry in entries:
            if entry["trade_id"] == trade_id:
                entry["exit_time"] = exit_time.isoformat()
                entry["exit_price"] = exit_price
                entry["exit_quantity"] = exit_quantity
                entry["exit_reason"] = exit_reason

                # Calculate PnL
                pnl_dollars = (exit_price - entry["entry_price"]) * exit_quantity
                if entry["direction"] == "short":
                    pnl_dollars = -pnl_dollars

                entry["pnl_dollars"] = pnl_dollars
                entry["pnl_percent"] = (pnl_dollars / (entry["entry_price"] * entry["entry_quantity"])) * 100

                # Calculate R-multiple
                risk_per_share = abs(entry["entry_price"] - entry["stop_loss"])
                if risk_per_share > 0:
                    entry["r_multiple"] = (exit_price - entry["entry_price"]) / risk_per_share
                    if entry["direction"] == "short":
                        entry["r_multiple"] = -entry["r_multiple"]
                else:
                    entry["r_multiple"] = 0.0

                break

        await self._write_entries(entries, date_str)
        logger.info(f"Logged exit for trade {trade_id}")

    async def add_emotion_tag(self, trade_id: str, tag: str) -> None:
        """Add an emotion tag to a trade.

        Args:
            trade_id: The trade ID to update.
            tag: The emotion tag to add.
        """
        await self._update_entry_field(trade_id, "emotion_tag", tag)

    async def add_notes(self, trade_id: str, notes: str) -> None:
        """Add notes to a trade.

        Args:
            trade_id: The trade ID to update.
            notes: The notes to add.
        """
        await self._update_entry_field(trade_id, "notes", notes)

    async def get_entries_for_date(self, date: datetime) -> list[JournalEntry]:
        """Get all entries for a specific date.

        Args:
            date: The date to get entries for.

        Returns:
            List of journal entries for that date.
        """
        date_str = date.strftime("%Y-%m-%d")
        entries_data = await self._load_entries(date_str)
        return [self._dict_to_entry(e) for e in entries_data]

    async def get_entries_for_period(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[JournalEntry]:
        """Get all entries within a date range.

        Args:
            start_date: Start of the period (inclusive).
            end_date: End of the period (inclusive).

        Returns:
            List of journal entries within the period.
        """
        entries = []
        current = start_date

        while current <= end_date:
            day_entries = await self.get_entries_for_date(current)
            entries.extend(day_entries)
            current = datetime(current.year, current.month, current.day + 1)

        return entries

    def _generate_trade_id(self, date_str: str, symbol: str) -> str:
        """Generate a unique trade ID.

        Args:
            date_str: The date string (YYYY-MM-DD).
            symbol: The stock symbol.

        Returns:
            Unique trade ID in format YYYY-MM-DD-SYMBOL-NNN.
        """
        if date_str not in self._trade_counts:
            self._trade_counts[date_str] = 0

        self._trade_counts[date_str] += 1
        count = self._trade_counts[date_str]

        return f"{date_str}-{symbol}-{count:03d}"

    def _determine_market_conditions(self, recommendation: TradeRecommendation) -> str:
        """Determine market conditions from recommendation.

        Args:
            recommendation: The trade recommendation.

        Returns:
            String describing market conditions.
        """
        conditions = []

        if recommendation.components.technical_score >= 70:
            conditions.append("bullish_trend")
        elif recommendation.components.technical_score <= 30:
            conditions.append("bearish_trend")
        else:
            conditions.append("neutral_trend")

        if recommendation.components.sentiment_score >= 70:
            conditions.append("high_sentiment")
        elif recommendation.components.sentiment_score <= 30:
            conditions.append("low_sentiment")

        return "_".join(conditions) if conditions else "unknown"

    async def _save_entry(self, entry: JournalEntry, date_str: str) -> None:
        """Save a new entry to the daily file.

        Args:
            entry: The journal entry to save.
            date_str: The date string for the file.
        """
        entries = await self._load_entries(date_str)
        entries.append(self._entry_to_dict(entry))
        await self._write_entries(entries, date_str)

    async def _load_entries(self, date_str: str) -> list[dict]:
        """Load entries from a daily file.

        Args:
            date_str: The date string for the file.

        Returns:
            List of entry dictionaries.
        """
        file_path = self._get_file_path(date_str)

        if not file_path.exists():
            return []

        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def _write_entries(self, entries: list[dict], date_str: str) -> None:
        """Write entries to a daily file.

        Args:
            entries: List of entry dictionaries.
            date_str: The date string for the file.
        """
        file_path = self._get_file_path(date_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(entries, indent=2, default=str))

    async def _update_entry_field(self, trade_id: str, field: str, value: str) -> None:
        """Update a single field in an entry.

        Args:
            trade_id: The trade ID to update.
            field: The field name to update.
            value: The new value.
        """
        date_str = "-".join(trade_id.split("-")[:3])
        entries = await self._load_entries(date_str)

        for entry in entries:
            if entry["trade_id"] == trade_id:
                entry[field] = value
                break

        await self._write_entries(entries, date_str)

    def _get_file_path(self, date_str: str) -> Path:
        """Get the file path for a date.

        Args:
            date_str: The date string (YYYY-MM-DD).

        Returns:
            Path to the JSON file.
        """
        return self._data_dir / f"{date_str}.json"

    def _entry_to_dict(self, entry: JournalEntry) -> dict:
        """Convert a JournalEntry to a dictionary.

        Args:
            entry: The journal entry.

        Returns:
            Dictionary representation.
        """
        return {
            "trade_id": entry.trade_id,
            "symbol": entry.symbol,
            "direction": entry.direction.value,
            "entry_time": entry.entry_time.isoformat(),
            "entry_price": entry.entry_price,
            "entry_quantity": entry.entry_quantity,
            "entry_reason": entry.entry_reason,
            "entry_score": entry.entry_score,
            "stop_loss": entry.stop_loss,
            "exit_time": entry.exit_time.isoformat() if entry.exit_time else None,
            "exit_price": entry.exit_price,
            "exit_quantity": entry.exit_quantity,
            "exit_reason": entry.exit_reason,
            "pnl_dollars": entry.pnl_dollars,
            "pnl_percent": entry.pnl_percent,
            "r_multiple": entry.r_multiple,
            "market_conditions": entry.market_conditions,
            "score_components": {
                "sentiment_score": entry.score_components.sentiment_score,
                "technical_score": entry.score_components.technical_score,
                "sentiment_weight": entry.score_components.sentiment_weight,
                "technical_weight": entry.score_components.technical_weight,
                "confluence_bonus": entry.score_components.confluence_bonus,
                "credibility_multiplier": entry.score_components.credibility_multiplier,
                "time_factor": entry.score_components.time_factor,
            },
            "emotion_tag": entry.emotion_tag,
            "notes": entry.notes,
        }

    def _dict_to_entry(self, data: dict) -> JournalEntry:
        """Convert a dictionary to a JournalEntry.

        Args:
            data: Dictionary representation.

        Returns:
            JournalEntry object.
        """
        from src.scoring.models import Direction, ScoreComponents

        return JournalEntry(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            direction=Direction(data["direction"]),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            entry_price=data["entry_price"],
            entry_quantity=data["entry_quantity"],
            entry_reason=data["entry_reason"],
            entry_score=data["entry_score"],
            stop_loss=data["stop_loss"],
            exit_time=datetime.fromisoformat(data["exit_time"]) if data["exit_time"] else None,
            exit_price=data["exit_price"],
            exit_quantity=data["exit_quantity"],
            exit_reason=data["exit_reason"],
            pnl_dollars=data["pnl_dollars"],
            pnl_percent=data["pnl_percent"],
            r_multiple=data["r_multiple"],
            market_conditions=data["market_conditions"],
            score_components=ScoreComponents(**data["score_components"]),
            emotion_tag=data["emotion_tag"],
            notes=data["notes"],
        )
```

Update `src/journal/__init__.py`:

```python
# src/journal/__init__.py
"""Journal module for trading history and metrics."""

from .models import JournalEntry, PatternAnalysis, TradingMetrics
from .settings import JournalSettings
from .trade_logger import TradeLogger

__all__ = [
    "JournalEntry",
    "JournalSettings",
    "PatternAnalysis",
    "TradeLogger",
    "TradingMetrics",
]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_trade_logger.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/journal/trade_logger.py tests/journal/test_trade_logger.py src/journal/__init__.py
git commit -m "feat(journal): add TradeLogger for persisting trades to JSON"
```

---

## Task 4: Create MetricsCalculator

**Files:**
- Create: `src/journal/metrics_calculator.py`
- Create: `tests/journal/test_metrics_calculator.py`
- Modify: `src/journal/__init__.py`

**Step 1: Write the failing test**

```python
# tests/journal/test_metrics_calculator.py
"""Tests for metrics calculator."""
import pytest
from datetime import datetime

from src.journal.metrics_calculator import MetricsCalculator
from src.journal.models import JournalEntry, TradingMetrics
from src.scoring.models import Direction, ScoreComponents


@pytest.fixture
def sample_components():
    """Create sample score components."""
    return ScoreComponents(
        sentiment_score=80.0,
        technical_score=75.0,
        sentiment_weight=0.5,
        technical_weight=0.5,
        confluence_bonus=0.1,
        credibility_multiplier=1.0,
        time_factor=1.0,
    )


@pytest.fixture
def winning_entry(sample_components):
    """Create a winning trade entry."""
    return JournalEntry(
        trade_id="2026-01-17-NVDA-001",
        symbol="NVDA",
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 17, 9, 33, 0),
        entry_price=140.0,
        entry_quantity=100,
        entry_reason="test",
        entry_score=85.0,
        stop_loss=138.0,
        exit_time=datetime(2026, 1, 17, 11, 0, 0),
        exit_price=144.0,
        exit_quantity=100,
        exit_reason="take_profit",
        pnl_dollars=400.0,
        pnl_percent=2.86,
        r_multiple=2.0,
        market_conditions="bullish",
        score_components=sample_components,
        emotion_tag=None,
        notes=None,
    )


@pytest.fixture
def losing_entry(sample_components):
    """Create a losing trade entry."""
    return JournalEntry(
        trade_id="2026-01-17-AAPL-002",
        symbol="AAPL",
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 17, 10, 0, 0),
        entry_price=180.0,
        entry_quantity=50,
        entry_reason="test",
        entry_score=70.0,
        stop_loss=178.0,
        exit_time=datetime(2026, 1, 17, 10, 30, 0),
        exit_price=178.0,
        exit_quantity=50,
        exit_reason="stop_loss",
        pnl_dollars=-100.0,
        pnl_percent=-1.11,
        r_multiple=-1.0,
        market_conditions="neutral",
        score_components=sample_components,
        emotion_tag=None,
        notes=None,
    )


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculate_with_no_entries(self):
        """calculate should return zeros for empty list."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate([], period_days=30)

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0

    def test_calculate_win_rate(self, winning_entry, losing_entry):
        """calculate should compute correct win rate."""
        calculator = MetricsCalculator()
        entries = [winning_entry, losing_entry]
        metrics = calculator.calculate(entries, period_days=30)

        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.5

    def test_calculate_profit_factor(self, winning_entry, losing_entry):
        """calculate should compute correct profit factor."""
        calculator = MetricsCalculator()
        entries = [winning_entry, losing_entry]
        metrics = calculator.calculate(entries, period_days=30)

        # Profit factor = gross_profit / gross_loss = 400 / 100 = 4.0
        assert metrics.profit_factor == 4.0

    def test_calculate_expectancy(self, winning_entry, losing_entry):
        """calculate should compute correct expectancy."""
        calculator = MetricsCalculator()
        entries = [winning_entry, losing_entry]
        metrics = calculator.calculate(entries, period_days=30)

        # Expectancy = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)
        # = (0.5 * 2.0) - (0.5 * 1.0) = 0.5
        assert metrics.expectancy == pytest.approx(0.5, rel=0.01)

    def test_calculate_total_pnl(self, winning_entry, losing_entry):
        """calculate should compute correct total PnL."""
        calculator = MetricsCalculator()
        entries = [winning_entry, losing_entry]
        metrics = calculator.calculate(entries, period_days=30)

        assert metrics.total_pnl_dollars == 300.0  # 400 - 100

    def test_calculate_best_worst_trades(self, winning_entry, losing_entry):
        """calculate should identify best and worst trades."""
        calculator = MetricsCalculator()
        entries = [winning_entry, losing_entry]
        metrics = calculator.calculate(entries, period_days=30)

        assert metrics.best_trade == winning_entry
        assert metrics.worst_trade == losing_entry

    def test_calculate_avg_win_loss(self, winning_entry, losing_entry):
        """calculate should compute average win and loss."""
        calculator = MetricsCalculator()
        entries = [winning_entry, losing_entry]
        metrics = calculator.calculate(entries, period_days=30)

        assert metrics.avg_win_dollars == 400.0
        assert metrics.avg_loss_dollars == 100.0
        assert metrics.avg_win_r == 2.0
        assert metrics.avg_loss_r == 1.0

    def test_ignores_open_trades(self, winning_entry, sample_components):
        """calculate should ignore trades without exits."""
        open_entry = JournalEntry(
            trade_id="2026-01-17-TSLA-003",
            symbol="TSLA",
            direction=Direction.LONG,
            entry_time=datetime(2026, 1, 17, 14, 0, 0),
            entry_price=200.0,
            entry_quantity=25,
            entry_reason="test",
            entry_score=75.0,
            stop_loss=195.0,
            exit_time=None,  # Still open
            exit_price=None,
            exit_quantity=0,
            exit_reason=None,
            pnl_dollars=0.0,
            pnl_percent=0.0,
            r_multiple=0.0,
            market_conditions="bullish",
            score_components=sample_components,
            emotion_tag=None,
            notes=None,
        )

        calculator = MetricsCalculator()
        metrics = calculator.calculate([winning_entry, open_entry], period_days=30)

        assert metrics.total_trades == 1  # Only closed trades
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_metrics_calculator.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/journal/metrics_calculator.py
"""Calculator for trading performance metrics."""
import math
from .models import JournalEntry, TradingMetrics


class MetricsCalculator:
    """Calculates trading performance metrics from journal entries."""

    def calculate(
        self,
        entries: list[JournalEntry],
        period_days: int = 30,
    ) -> TradingMetrics:
        """Calculate trading metrics from journal entries.

        Args:
            entries: List of journal entries to analyze.
            period_days: Number of days in the period.

        Returns:
            TradingMetrics with calculated values.
        """
        # Filter to closed trades only
        closed = [e for e in entries if not e.is_open]

        if not closed:
            return self._empty_metrics(period_days)

        # Separate winners and losers
        winners = [e for e in closed if e.pnl_dollars > 0]
        losers = [e for e in closed if e.pnl_dollars <= 0]

        total_trades = len(closed)
        winning_trades = len(winners)
        losing_trades = len(losers)

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Gross profit and loss
        gross_profit = sum(e.pnl_dollars for e in winners)
        gross_loss = abs(sum(e.pnl_dollars for e in losers))

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Average win/loss
        avg_win_dollars = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss_dollars = gross_loss / losing_trades if losing_trades > 0 else 0.0

        avg_win_r = sum(e.r_multiple for e in winners) / winning_trades if winning_trades > 0 else 0.0
        avg_loss_r = abs(sum(e.r_multiple for e in losers)) / losing_trades if losing_trades > 0 else 0.0

        # Expectancy
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

        # Total PnL
        total_pnl_dollars = sum(e.pnl_dollars for e in closed)
        total_entry_value = sum(e.entry_price * e.entry_quantity for e in closed)
        total_pnl_percent = (total_pnl_dollars / total_entry_value * 100) if total_entry_value > 0 else 0.0

        # Max drawdown
        max_drawdown_percent = self._calculate_max_drawdown(closed)

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(closed)

        # Best and worst trades
        best_trade = max(closed, key=lambda e: e.pnl_dollars)
        worst_trade = min(closed, key=lambda e: e.pnl_dollars)

        return TradingMetrics(
            period_days=period_days,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win_dollars=avg_win_dollars,
            avg_loss_dollars=avg_loss_dollars,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            total_pnl_dollars=total_pnl_dollars,
            total_pnl_percent=total_pnl_percent,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )

    def _empty_metrics(self, period_days: int) -> TradingMetrics:
        """Return empty metrics when no trades exist."""
        return TradingMetrics(
            period_days=period_days,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_win_dollars=0.0,
            avg_loss_dollars=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            total_pnl_dollars=0.0,
            total_pnl_percent=0.0,
            max_drawdown_percent=0.0,
            sharpe_ratio=0.0,
            best_trade=None,
            worst_trade=None,
        )

    def _calculate_max_drawdown(self, entries: list[JournalEntry]) -> float:
        """Calculate maximum drawdown percentage.

        Args:
            entries: Closed trade entries sorted by exit time.

        Returns:
            Maximum drawdown as a percentage.
        """
        if not entries:
            return 0.0

        # Sort by exit time
        sorted_entries = sorted(entries, key=lambda e: e.exit_time or e.entry_time)

        # Calculate cumulative PnL
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for entry in sorted_entries:
            cumulative += entry.pnl_dollars
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Convert to percentage (relative to peak)
        return (max_drawdown / peak * 100) if peak > 0 else 0.0

    def _calculate_sharpe_ratio(self, entries: list[JournalEntry]) -> float:
        """Calculate annualized Sharpe ratio.

        Args:
            entries: Closed trade entries.

        Returns:
            Annualized Sharpe ratio.
        """
        if len(entries) < 2:
            return 0.0

        returns = [e.pnl_percent for e in entries]
        avg_return = sum(returns) / len(returns)

        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Annualize assuming 252 trading days
        sharpe = (avg_return / std_dev) * math.sqrt(252)
        return sharpe
```

Update `src/journal/__init__.py`:

```python
# src/journal/__init__.py
"""Journal module for trading history and metrics."""

from .metrics_calculator import MetricsCalculator
from .models import JournalEntry, PatternAnalysis, TradingMetrics
from .settings import JournalSettings
from .trade_logger import TradeLogger

__all__ = [
    "JournalEntry",
    "JournalSettings",
    "MetricsCalculator",
    "PatternAnalysis",
    "TradeLogger",
    "TradingMetrics",
]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_metrics_calculator.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/journal/metrics_calculator.py tests/journal/test_metrics_calculator.py src/journal/__init__.py
git commit -m "feat(journal): add MetricsCalculator for trading statistics"
```

---

## Task 5: Create PatternAnalyzer

**Files:**
- Create: `src/journal/pattern_analyzer.py`
- Create: `tests/journal/test_pattern_analyzer.py`
- Modify: `src/journal/__init__.py`

**Step 1: Write the failing test**

```python
# tests/journal/test_pattern_analyzer.py
"""Tests for pattern analyzer."""
import pytest
from datetime import datetime

from src.journal.pattern_analyzer import PatternAnalyzer
from src.journal.models import JournalEntry, PatternAnalysis
from src.scoring.models import Direction, ScoreComponents


@pytest.fixture
def sample_components():
    """Create sample score components."""
    return ScoreComponents(
        sentiment_score=80.0,
        technical_score=75.0,
        sentiment_weight=0.5,
        technical_weight=0.5,
        confluence_bonus=0.1,
        credibility_multiplier=1.0,
        time_factor=1.0,
    )


@pytest.fixture
def morning_winner(sample_components):
    """Trade at 10 AM that wins."""
    return JournalEntry(
        trade_id="2026-01-17-NVDA-001",
        symbol="NVDA",
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 17, 10, 0, 0),  # 10 AM, Friday
        entry_price=140.0,
        entry_quantity=100,
        entry_reason="unusual_whales_sweep",
        entry_score=85.0,
        stop_loss=138.0,
        exit_time=datetime(2026, 1, 17, 10, 30, 0),
        exit_price=144.0,
        exit_quantity=100,
        exit_reason="take_profit",
        pnl_dollars=400.0,
        pnl_percent=2.86,
        r_multiple=2.0,
        market_conditions="bullish",
        score_components=sample_components,
        emotion_tag=None,
        notes=None,
    )


@pytest.fixture
def afternoon_loser(sample_components):
    """Trade at 2 PM that loses."""
    return JournalEntry(
        trade_id="2026-01-17-AAPL-002",
        symbol="AAPL",
        direction=Direction.LONG,
        entry_time=datetime(2026, 1, 17, 14, 0, 0),  # 2 PM, Friday
        entry_price=180.0,
        entry_quantity=50,
        entry_reason="momentum_breakout",
        entry_score=70.0,
        stop_loss=178.0,
        exit_time=datetime(2026, 1, 17, 15, 0, 0),
        exit_price=178.0,
        exit_quantity=50,
        exit_reason="stop_loss",
        pnl_dollars=-100.0,
        pnl_percent=-1.11,
        r_multiple=-1.0,
        market_conditions="neutral",
        score_components=sample_components,
        emotion_tag=None,
        notes=None,
    )


class TestPatternAnalyzer:
    """Tests for PatternAnalyzer."""

    def test_analyze_empty_entries(self):
        """analyze should return defaults for empty list."""
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze([])

        assert analysis.best_hour == -1
        assert analysis.worst_hour == -1
        assert analysis.best_symbols == []
        assert analysis.worst_symbols == []

    def test_analyze_best_worst_hour(self, morning_winner, afternoon_loser):
        """analyze should identify best and worst trading hours."""
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze([morning_winner, afternoon_loser])

        assert analysis.best_hour == 10  # Morning trade won
        assert analysis.worst_hour == 14  # Afternoon trade lost

    def test_analyze_best_day_of_week(self, morning_winner, afternoon_loser):
        """analyze should identify best day of week."""
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze([morning_winner, afternoon_loser])

        # Both trades on Friday (4), net positive so it's the best
        assert analysis.best_day_of_week == 4

    def test_analyze_best_worst_symbols(self, morning_winner, afternoon_loser):
        """analyze should rank symbols by average PnL."""
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze([morning_winner, afternoon_loser])

        assert analysis.best_symbols[0] == ("NVDA", 400.0)
        assert analysis.worst_symbols[0] == ("AAPL", -100.0)

    def test_analyze_best_worst_setups(self, morning_winner, afternoon_loser):
        """analyze should rank setups by average PnL."""
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze([morning_winner, afternoon_loser])

        assert analysis.best_setups[0] == ("unusual_whales_sweep", 400.0)
        assert analysis.worst_setups[0] == ("momentum_breakout", -100.0)

    def test_analyze_duration(self, morning_winner, afternoon_loser):
        """analyze should calculate average duration for winners vs losers."""
        analyzer = PatternAnalyzer()
        analysis = analyzer.analyze([morning_winner, afternoon_loser])

        # Morning winner: 30 minutes
        assert analysis.avg_winner_duration_minutes == 30.0
        # Afternoon loser: 60 minutes
        assert analysis.avg_loser_duration_minutes == 60.0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_pattern_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/journal/pattern_analyzer.py
"""Analyzer for trading patterns."""
from collections import defaultdict

from .models import JournalEntry, PatternAnalysis


class PatternAnalyzer:
    """Analyzes trading patterns from journal entries."""

    def analyze(self, entries: list[JournalEntry]) -> PatternAnalysis:
        """Analyze trading patterns from journal entries.

        Args:
            entries: List of journal entries to analyze.

        Returns:
            PatternAnalysis with identified patterns.
        """
        # Filter to closed trades only
        closed = [e for e in entries if not e.is_open]

        if not closed:
            return self._empty_analysis()

        # Analyze by hour
        best_hour, worst_hour = self._analyze_by_hour(closed)

        # Analyze by day of week
        best_day = self._analyze_by_day(closed)

        # Analyze by symbol
        best_symbols, worst_symbols = self._analyze_by_symbol(closed)

        # Analyze by setup (entry reason)
        best_setups, worst_setups = self._analyze_by_setup(closed)

        # Analyze duration
        avg_winner_duration, avg_loser_duration = self._analyze_duration(closed)

        return PatternAnalysis(
            best_hour=best_hour,
            worst_hour=worst_hour,
            best_day_of_week=best_day,
            best_symbols=best_symbols,
            worst_symbols=worst_symbols,
            best_setups=best_setups,
            worst_setups=worst_setups,
            avg_winner_duration_minutes=avg_winner_duration,
            avg_loser_duration_minutes=avg_loser_duration,
        )

    def _empty_analysis(self) -> PatternAnalysis:
        """Return empty analysis when no trades exist."""
        return PatternAnalysis(
            best_hour=-1,
            worst_hour=-1,
            best_day_of_week=-1,
            best_symbols=[],
            worst_symbols=[],
            best_setups=[],
            worst_setups=[],
            avg_winner_duration_minutes=0.0,
            avg_loser_duration_minutes=0.0,
        )

    def _analyze_by_hour(self, entries: list[JournalEntry]) -> tuple[int, int]:
        """Find best and worst trading hours.

        Args:
            entries: Closed trade entries.

        Returns:
            Tuple of (best_hour, worst_hour).
        """
        hour_pnl: dict[int, list[float]] = defaultdict(list)

        for entry in entries:
            hour = entry.entry_time.hour
            hour_pnl[hour].append(entry.pnl_dollars)

        # Calculate average PnL per hour
        hour_avg = {h: sum(pnls) / len(pnls) for h, pnls in hour_pnl.items()}

        best_hour = max(hour_avg, key=hour_avg.get)
        worst_hour = min(hour_avg, key=hour_avg.get)

        return best_hour, worst_hour

    def _analyze_by_day(self, entries: list[JournalEntry]) -> int:
        """Find best trading day of week.

        Args:
            entries: Closed trade entries.

        Returns:
            Best day of week (0=Monday, 4=Friday).
        """
        day_pnl: dict[int, list[float]] = defaultdict(list)

        for entry in entries:
            day = entry.entry_time.weekday()
            day_pnl[day].append(entry.pnl_dollars)

        # Calculate average PnL per day
        day_avg = {d: sum(pnls) / len(pnls) for d, pnls in day_pnl.items()}

        return max(day_avg, key=day_avg.get)

    def _analyze_by_symbol(
        self, entries: list[JournalEntry]
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Rank symbols by average PnL.

        Args:
            entries: Closed trade entries.

        Returns:
            Tuple of (best_symbols, worst_symbols) with avg PnL.
        """
        symbol_pnl: dict[str, list[float]] = defaultdict(list)

        for entry in entries:
            symbol_pnl[entry.symbol].append(entry.pnl_dollars)

        # Calculate average PnL per symbol
        symbol_avg = [(s, sum(pnls) / len(pnls)) for s, pnls in symbol_pnl.items()]

        # Sort by average PnL
        sorted_symbols = sorted(symbol_avg, key=lambda x: x[1], reverse=True)

        best = [(s, avg) for s, avg in sorted_symbols if avg > 0]
        worst = [(s, avg) for s, avg in sorted_symbols if avg <= 0]

        return best[:5], worst[:5]  # Top/bottom 5

    def _analyze_by_setup(
        self, entries: list[JournalEntry]
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Rank setups (entry reasons) by average PnL.

        Args:
            entries: Closed trade entries.

        Returns:
            Tuple of (best_setups, worst_setups) with avg PnL.
        """
        setup_pnl: dict[str, list[float]] = defaultdict(list)

        for entry in entries:
            setup_pnl[entry.entry_reason].append(entry.pnl_dollars)

        # Calculate average PnL per setup
        setup_avg = [(s, sum(pnls) / len(pnls)) for s, pnls in setup_pnl.items()]

        # Sort by average PnL
        sorted_setups = sorted(setup_avg, key=lambda x: x[1], reverse=True)

        best = [(s, avg) for s, avg in sorted_setups if avg > 0]
        worst = [(s, avg) for s, avg in sorted_setups if avg <= 0]

        return best[:5], worst[:5]

    def _analyze_duration(
        self, entries: list[JournalEntry]
    ) -> tuple[float, float]:
        """Calculate average duration for winners vs losers.

        Args:
            entries: Closed trade entries.

        Returns:
            Tuple of (avg_winner_minutes, avg_loser_minutes).
        """
        winner_durations = []
        loser_durations = []

        for entry in entries:
            if entry.exit_time is None:
                continue

            duration = (entry.exit_time - entry.entry_time).total_seconds() / 60

            if entry.pnl_dollars > 0:
                winner_durations.append(duration)
            else:
                loser_durations.append(duration)

        avg_winner = sum(winner_durations) / len(winner_durations) if winner_durations else 0.0
        avg_loser = sum(loser_durations) / len(loser_durations) if loser_durations else 0.0

        return avg_winner, avg_loser
```

Update `src/journal/__init__.py`:

```python
# src/journal/__init__.py
"""Journal module for trading history and metrics."""

from .metrics_calculator import MetricsCalculator
from .models import JournalEntry, PatternAnalysis, TradingMetrics
from .pattern_analyzer import PatternAnalyzer
from .settings import JournalSettings
from .trade_logger import TradeLogger

__all__ = [
    "JournalEntry",
    "JournalSettings",
    "MetricsCalculator",
    "PatternAnalysis",
    "PatternAnalyzer",
    "TradeLogger",
    "TradingMetrics",
]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_pattern_analyzer.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/journal/pattern_analyzer.py tests/journal/test_pattern_analyzer.py src/journal/__init__.py
git commit -m "feat(journal): add PatternAnalyzer for trading insights"
```

---

## Task 6: Create JournalManager

**Files:**
- Create: `src/journal/journal_manager.py`
- Create: `tests/journal/test_journal_manager.py`
- Modify: `src/journal/__init__.py`

**Step 1: Write the failing test**

```python
# tests/journal/test_journal_manager.py
"""Tests for journal manager."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.journal.journal_manager import JournalManager
from src.journal.settings import JournalSettings
from src.journal.models import JournalEntry, TradingMetrics, PatternAnalysis
from src.scoring.models import Direction, ScoreComponents, TradeRecommendation, ScoreTier
from src.execution.models import TrackedPosition


@pytest.fixture
def settings(tmp_path):
    """Create settings with temp directory."""
    return JournalSettings(data_dir=str(tmp_path / "trades"))


@pytest.fixture
def journal_manager(settings):
    """Create a JournalManager instance."""
    return JournalManager(settings)


@pytest.fixture
def sample_position():
    """Create a sample tracked position."""
    return TrackedPosition(
        symbol="NVDA",
        quantity=322,
        entry_price=140.76,
        entry_time=datetime(2026, 1, 17, 9, 33, 0),
        stop_loss=139.20,
        take_profit=145.40,
        order_id="order-123",
        direction=Direction.LONG,
    )


@pytest.fixture
def sample_recommendation():
    """Create a sample trade recommendation."""
    components = ScoreComponents(
        sentiment_score=80.0,
        technical_score=75.0,
        sentiment_weight=0.5,
        technical_weight=0.5,
        confluence_bonus=0.1,
        credibility_multiplier=1.0,
        time_factor=1.0,
    )
    return TradeRecommendation(
        symbol="NVDA",
        direction=Direction.LONG,
        score=88.0,
        tier=ScoreTier.STRONG,
        position_size_percent=90.0,
        entry_price=140.76,
        stop_loss=139.20,
        take_profit=145.40,
        risk_reward_ratio=3.0,
        components=components,
        reasoning="unusual_whales_sweep_alert",
        timestamp=datetime(2026, 1, 17, 9, 33, 0),
    )


class TestJournalManager:
    """Tests for JournalManager."""

    @pytest.mark.asyncio
    async def test_on_trade_opened(self, journal_manager, sample_position, sample_recommendation):
        """on_trade_opened should log the entry."""
        trade_id = await journal_manager.on_trade_opened(sample_position, sample_recommendation)

        assert trade_id == "2026-01-17-NVDA-001"

    @pytest.mark.asyncio
    async def test_on_trade_closed(self, journal_manager, sample_position, sample_recommendation):
        """on_trade_closed should log the exit."""
        trade_id = await journal_manager.on_trade_opened(sample_position, sample_recommendation)

        await journal_manager.on_trade_closed(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 11, 45, 0),
            exit_price=142.38,
            exit_quantity=322,
            exit_reason="trailing_stop",
        )

        # Verify entry was updated
        entries = await journal_manager.get_entries_for_date(datetime(2026, 1, 17))
        assert len(entries) == 1
        assert entries[0].exit_price == 142.38

    @pytest.mark.asyncio
    async def test_get_daily_summary(self, journal_manager, sample_position, sample_recommendation):
        """get_daily_summary should return metrics for today."""
        trade_id = await journal_manager.on_trade_opened(sample_position, sample_recommendation)
        await journal_manager.on_trade_closed(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 11, 45, 0),
            exit_price=142.38,
            exit_quantity=322,
            exit_reason="trailing_stop",
        )

        summary = await journal_manager.get_daily_summary(datetime(2026, 1, 17))

        assert summary["total_trades"] == 1
        assert summary["total_pnl"] > 0

    @pytest.mark.asyncio
    async def test_get_weekly_report(self, journal_manager, sample_position, sample_recommendation):
        """get_weekly_report should return metrics and patterns."""
        trade_id = await journal_manager.on_trade_opened(sample_position, sample_recommendation)
        await journal_manager.on_trade_closed(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 11, 45, 0),
            exit_price=142.38,
            exit_quantity=322,
            exit_reason="trailing_stop",
        )

        report = await journal_manager.get_weekly_report(datetime(2026, 1, 17))

        assert "metrics" in report
        assert "patterns" in report
        assert report["metrics"].total_trades == 1

    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self, tmp_path, sample_position, sample_recommendation):
        """When disabled, operations should be no-ops."""
        settings = JournalSettings(
            enabled=False,
            data_dir=str(tmp_path / "trades"),
        )
        manager = JournalManager(settings)

        trade_id = await manager.on_trade_opened(sample_position, sample_recommendation)

        assert trade_id is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_journal_manager.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/journal/journal_manager.py
"""Manager for the trading journal."""
import logging
from datetime import datetime, timedelta

from src.execution.models import TrackedPosition
from src.scoring.models import TradeRecommendation

from .metrics_calculator import MetricsCalculator
from .models import JournalEntry, PatternAnalysis, TradingMetrics
from .pattern_analyzer import PatternAnalyzer
from .settings import JournalSettings
from .trade_logger import TradeLogger

logger = logging.getLogger(__name__)


class JournalManager:
    """Orchestrates journal components.

    Provides high-level interface for logging trades and generating reports.
    """

    def __init__(self, settings: JournalSettings) -> None:
        """Initialize the journal manager.

        Args:
            settings: Journal configuration settings.
        """
        self._settings = settings
        self._logger = TradeLogger(settings)
        self._metrics_calculator = MetricsCalculator()
        self._pattern_analyzer = PatternAnalyzer()

    async def on_trade_opened(
        self,
        position: TrackedPosition,
        recommendation: TradeRecommendation,
    ) -> str | None:
        """Handle a trade being opened.

        Args:
            position: The position that was opened.
            recommendation: The recommendation that triggered the trade.

        Returns:
            The trade ID, or None if journaling is disabled.
        """
        if not self._settings.enabled or not self._settings.auto_log_entries:
            return None

        trade_id = await self._logger.log_entry(position, recommendation)
        logger.info(f"Journal: Trade opened - {trade_id}")
        return trade_id

    async def on_trade_closed(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        exit_quantity: int,
        exit_reason: str,
    ) -> None:
        """Handle a trade being closed.

        Args:
            trade_id: The trade ID to update.
            exit_time: When the trade was closed.
            exit_price: The exit price.
            exit_quantity: Number of shares exited.
            exit_reason: Reason for the exit.
        """
        if not self._settings.enabled or not self._settings.auto_log_exits:
            return

        await self._logger.log_exit(
            trade_id=trade_id,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_quantity=exit_quantity,
            exit_reason=exit_reason,
        )
        logger.info(f"Journal: Trade closed - {trade_id}")

    async def add_emotion_tag(self, trade_id: str, tag: str) -> None:
        """Add an emotion tag to a trade.

        Args:
            trade_id: The trade ID.
            tag: The emotion tag.
        """
        if not self._settings.enabled:
            return

        await self._logger.add_emotion_tag(trade_id, tag)

    async def add_notes(self, trade_id: str, notes: str) -> None:
        """Add notes to a trade.

        Args:
            trade_id: The trade ID.
            notes: The notes to add.
        """
        if not self._settings.enabled:
            return

        await self._logger.add_notes(trade_id, notes)

    async def get_entries_for_date(self, date: datetime) -> list[JournalEntry]:
        """Get all entries for a specific date.

        Args:
            date: The date to get entries for.

        Returns:
            List of journal entries.
        """
        return await self._logger.get_entries_for_date(date)

    async def get_daily_summary(self, date: datetime) -> dict:
        """Get a summary for a single day.

        Args:
            date: The date to summarize.

        Returns:
            Dictionary with summary data for Telegram.
        """
        entries = await self._logger.get_entries_for_date(date)
        closed = [e for e in entries if not e.is_open]

        if not closed:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
            }

        winners = [e for e in closed if e.pnl_dollars > 0]
        losers = [e for e in closed if e.pnl_dollars <= 0]
        total_pnl = sum(e.pnl_dollars for e in closed)

        return {
            "date": date.strftime("%Y-%m-%d"),
            "total_trades": len(closed),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "total_pnl": total_pnl,
            "win_rate": len(winners) / len(closed) if closed else 0.0,
        }

    async def get_weekly_report(self, end_date: datetime) -> dict:
        """Get a weekly report.

        Args:
            end_date: The end date of the week.

        Returns:
            Dictionary with metrics and patterns.
        """
        start_date = end_date - timedelta(days=7)

        entries = await self._logger.get_entries_for_period(start_date, end_date)

        metrics = self._metrics_calculator.calculate(entries, period_days=7)
        patterns = self._pattern_analyzer.analyze(entries)

        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "metrics": metrics,
            "patterns": patterns,
        }

    async def get_metrics(self, period_days: int = 30) -> TradingMetrics:
        """Get trading metrics for a period.

        Args:
            period_days: Number of days to include.

        Returns:
            TradingMetrics for the period.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        entries = await self._logger.get_entries_for_period(start_date, end_date)

        return self._metrics_calculator.calculate(entries, period_days)

    async def get_patterns(self, period_days: int = 30) -> PatternAnalysis:
        """Get pattern analysis for a period.

        Args:
            period_days: Number of days to include.

        Returns:
            PatternAnalysis for the period.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        entries = await self._logger.get_entries_for_period(start_date, end_date)

        return self._pattern_analyzer.analyze(entries)
```

Update `src/journal/__init__.py`:

```python
# src/journal/__init__.py
"""Journal module for trading history and metrics."""

from .journal_manager import JournalManager
from .metrics_calculator import MetricsCalculator
from .models import JournalEntry, PatternAnalysis, TradingMetrics
from .pattern_analyzer import PatternAnalyzer
from .settings import JournalSettings
from .trade_logger import TradeLogger

__all__ = [
    "JournalEntry",
    "JournalManager",
    "JournalSettings",
    "MetricsCalculator",
    "PatternAnalysis",
    "PatternAnalyzer",
    "TradeLogger",
    "TradingMetrics",
]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_journal_manager.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/journal/journal_manager.py tests/journal/test_journal_manager.py src/journal/__init__.py
git commit -m "feat(journal): add JournalManager to orchestrate components"
```

---

## Task 7: Update Config Settings

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`

**Step 1: Update settings.py**

Add to `src/config/settings.py`:

```python
# Add import at top
from src.journal.settings import JournalSettings

# Add to Settings class (around line 224)
journal: JournalSettings = Field(default_factory=JournalSettings)
```

**Step 2: Update settings.yaml**

Add journal section to `config/settings.yaml`:

```yaml
# Journal
journal:
  enabled: true
  data_dir: "data/trades"
  auto_log_entries: true
  auto_log_exits: true
  default_period_days: 30
  weekly_report_enabled: true
  weekly_report_day: "saturday"
```

**Step 3: Verify existing tests pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/config/ -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/config/settings.py config/settings.yaml
git commit -m "feat(config): add JournalSettings to system configuration"
```

---

## Task 8: Integration Tests

**Files:**
- Create: `tests/journal/test_integration.py`

**Step 1: Write integration test**

```python
# tests/journal/test_integration.py
"""Integration tests for journal module."""
import pytest
from datetime import datetime

from src.journal import (
    JournalManager,
    JournalSettings,
    TradeLogger,
    MetricsCalculator,
    PatternAnalyzer,
)
from src.scoring.models import Direction, ScoreComponents, TradeRecommendation, ScoreTier
from src.execution.models import TrackedPosition


@pytest.fixture
def settings(tmp_path):
    """Create settings with temp directory."""
    return JournalSettings(data_dir=str(tmp_path / "trades"))


@pytest.fixture
def sample_components():
    """Create sample score components."""
    return ScoreComponents(
        sentiment_score=80.0,
        technical_score=75.0,
        sentiment_weight=0.5,
        technical_weight=0.5,
        confluence_bonus=0.1,
        credibility_multiplier=1.0,
        time_factor=1.0,
    )


class TestJournalIntegration:
    """Integration tests for full journal flow."""

    @pytest.mark.asyncio
    async def test_full_trade_lifecycle(self, settings, sample_components):
        """Test complete trade: open -> close -> metrics -> patterns."""
        manager = JournalManager(settings)

        # Create position and recommendation
        position = TrackedPosition(
            symbol="NVDA",
            quantity=100,
            entry_price=140.0,
            entry_time=datetime(2026, 1, 17, 10, 0, 0),
            stop_loss=138.0,
            take_profit=146.0,
            order_id="order-123",
            direction=Direction.LONG,
        )
        recommendation = TradeRecommendation(
            symbol="NVDA",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=80.0,
            entry_price=140.0,
            stop_loss=138.0,
            take_profit=146.0,
            risk_reward_ratio=3.0,
            components=sample_components,
            reasoning="unusual_whales_sweep",
            timestamp=datetime(2026, 1, 17, 10, 0, 0),
        )

        # Open trade
        trade_id = await manager.on_trade_opened(position, recommendation)
        assert trade_id is not None

        # Close trade
        await manager.on_trade_closed(
            trade_id=trade_id,
            exit_time=datetime(2026, 1, 17, 11, 0, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="take_profit_1",
        )

        # Get daily summary
        summary = await manager.get_daily_summary(datetime(2026, 1, 17))
        assert summary["total_trades"] == 1
        assert summary["winning_trades"] == 1
        assert summary["total_pnl"] == 400.0

        # Get metrics
        metrics = await manager.get_metrics(period_days=30)
        assert metrics.total_trades == 1
        assert metrics.win_rate == 1.0

        # Get patterns
        patterns = await manager.get_patterns(period_days=30)
        assert patterns.best_hour == 10
        assert patterns.best_symbols[0][0] == "NVDA"

    @pytest.mark.asyncio
    async def test_multiple_trades_same_day(self, settings, sample_components):
        """Test multiple trades on same day."""
        manager = JournalManager(settings)

        # First trade - winner
        position1 = TrackedPosition(
            symbol="NVDA",
            quantity=100,
            entry_price=140.0,
            entry_time=datetime(2026, 1, 17, 10, 0, 0),
            stop_loss=138.0,
            take_profit=146.0,
            order_id="order-1",
            direction=Direction.LONG,
        )
        rec1 = TradeRecommendation(
            symbol="NVDA",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=80.0,
            entry_price=140.0,
            stop_loss=138.0,
            take_profit=146.0,
            risk_reward_ratio=3.0,
            components=sample_components,
            reasoning="setup_1",
            timestamp=datetime(2026, 1, 17, 10, 0, 0),
        )

        trade_id_1 = await manager.on_trade_opened(position1, rec1)
        await manager.on_trade_closed(
            trade_id=trade_id_1,
            exit_time=datetime(2026, 1, 17, 10, 30, 0),
            exit_price=144.0,
            exit_quantity=100,
            exit_reason="take_profit",
        )

        # Second trade - loser
        position2 = TrackedPosition(
            symbol="AAPL",
            quantity=50,
            entry_price=180.0,
            entry_time=datetime(2026, 1, 17, 14, 0, 0),
            stop_loss=178.0,
            take_profit=186.0,
            order_id="order-2",
            direction=Direction.LONG,
        )
        rec2 = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=70.0,
            tier=ScoreTier.MODERATE,
            position_size_percent=50.0,
            entry_price=180.0,
            stop_loss=178.0,
            take_profit=186.0,
            risk_reward_ratio=3.0,
            components=sample_components,
            reasoning="setup_2",
            timestamp=datetime(2026, 1, 17, 14, 0, 0),
        )

        trade_id_2 = await manager.on_trade_opened(position2, rec2)
        await manager.on_trade_closed(
            trade_id=trade_id_2,
            exit_time=datetime(2026, 1, 17, 14, 30, 0),
            exit_price=178.0,
            exit_quantity=50,
            exit_reason="stop_loss",
        )

        # Verify metrics
        summary = await manager.get_daily_summary(datetime(2026, 1, 17))
        assert summary["total_trades"] == 2
        assert summary["winning_trades"] == 1
        assert summary["losing_trades"] == 1
        assert summary["total_pnl"] == 300.0  # 400 - 100

    @pytest.mark.asyncio
    async def test_emotion_tags_and_notes(self, settings, sample_components):
        """Test adding emotion tags and notes to trades."""
        manager = JournalManager(settings)

        position = TrackedPosition(
            symbol="TSLA",
            quantity=25,
            entry_price=200.0,
            entry_time=datetime(2026, 1, 17, 11, 0, 0),
            stop_loss=195.0,
            take_profit=210.0,
            order_id="order-3",
            direction=Direction.LONG,
        )
        recommendation = TradeRecommendation(
            symbol="TSLA",
            direction=Direction.LONG,
            score=75.0,
            tier=ScoreTier.MODERATE,
            position_size_percent=60.0,
            entry_price=200.0,
            stop_loss=195.0,
            take_profit=210.0,
            risk_reward_ratio=2.0,
            components=sample_components,
            reasoning="momentum_play",
            timestamp=datetime(2026, 1, 17, 11, 0, 0),
        )

        trade_id = await manager.on_trade_opened(position, recommendation)

        # Add emotion tag and notes
        await manager.add_emotion_tag(trade_id, "confident")
        await manager.add_notes(trade_id, "Strong conviction, followed plan")

        # Verify
        entries = await manager.get_entries_for_date(datetime(2026, 1, 17))
        assert len(entries) == 1
        assert entries[0].emotion_tag == "confident"
        assert entries[0].notes == "Strong conviction, followed plan"
```

**Step 2: Run integration tests**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/test_integration.py -v`
Expected: PASS (3 tests)

**Step 3: Commit**

```bash
git add tests/journal/test_integration.py
git commit -m "test(journal): add integration tests for full trade lifecycle"
```

---

## Task 9: Final Test Suite Run

**Step 1: Run all journal tests**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/ -v`
Expected: All tests pass (~30 tests)

**Step 2: Run full test suite**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/ -v`
Expected: All tests pass

**Step 3: Check coverage**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && uv run pytest tests/journal/ --cov=src/journal --cov-report=term-missing`
Expected: >90% coverage

**Step 4: Verify all commits**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase10-journal && git log --oneline -10`
Expected: See all feature commits for Phase 10

---

*Plan complete. Ready for execution.*
