# Phase 5: Risk Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add simple risk management that gates trade execution based on max position size and daily loss limits.

**Architecture:** RiskManager validates each TradeRecommendation before execution, returning approved/blocked status. Tracks realized P&L to enforce daily limits, warns on unrealized drawdown.

**Tech Stack:** Python dataclasses, Pydantic settings, pytest

---

## Task 1: Data Models (RiskCheckResult, DailyRiskState)

**Files:**
- Create: `src/risk/__init__.py`
- Create: `src/risk/models.py`
- Create: `tests/risk/__init__.py`
- Create: `tests/risk/test_models.py`

**Step 1: Write the failing tests**

```python
# tests/risk/test_models.py
"""Tests for risk management data models."""

from datetime import date

from src.risk.models import RiskCheckResult, DailyRiskState


class TestRiskCheckResult:
    """Test suite for RiskCheckResult dataclass."""

    def test_approved_trade(self):
        """Test creating an approved trade result."""
        result = RiskCheckResult(
            approved=True,
            adjusted_quantity=100,
            adjusted_value=1000.0,
            rejection_reason=None,
            warnings=[],
        )
        assert result.approved is True
        assert result.adjusted_quantity == 100
        assert result.adjusted_value == 1000.0
        assert result.rejection_reason is None
        assert result.warnings == []

    def test_rejected_trade(self):
        """Test creating a rejected trade result."""
        result = RiskCheckResult(
            approved=False,
            adjusted_quantity=0,
            adjusted_value=0.0,
            rejection_reason="Daily loss limit exceeded",
            warnings=[],
        )
        assert result.approved is False
        assert result.adjusted_quantity == 0
        assert result.rejection_reason == "Daily loss limit exceeded"

    def test_approved_with_warnings(self):
        """Test approved trade with warnings."""
        result = RiskCheckResult(
            approved=True,
            adjusted_quantity=50,
            adjusted_value=500.0,
            rejection_reason=None,
            warnings=["Unrealized loss at $250, approaching threshold"],
        )
        assert result.approved is True
        assert len(result.warnings) == 1
        assert "Unrealized" in result.warnings[0]


class TestDailyRiskState:
    """Test suite for DailyRiskState dataclass."""

    def test_initial_state(self):
        """Test creating initial daily state."""
        state = DailyRiskState(
            date=date(2026, 1, 17),
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            trades_today=0,
            is_blocked=False,
        )
        assert state.date == date(2026, 1, 17)
        assert state.realized_pnl == 0.0
        assert state.unrealized_pnl == 0.0
        assert state.trades_today == 0
        assert state.is_blocked is False

    def test_state_with_losses(self):
        """Test state after losses."""
        state = DailyRiskState(
            date=date(2026, 1, 17),
            realized_pnl=-350.0,
            unrealized_pnl=-100.0,
            trades_today=5,
            is_blocked=False,
        )
        assert state.realized_pnl == -350.0
        assert state.unrealized_pnl == -100.0
        assert state.trades_today == 5

    def test_blocked_state(self):
        """Test blocked state when limit hit."""
        state = DailyRiskState(
            date=date(2026, 1, 17),
            realized_pnl=-500.0,
            unrealized_pnl=-50.0,
            trades_today=8,
            is_blocked=True,
        )
        assert state.is_blocked is True
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/risk/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.risk'"

**Step 3: Write minimal implementation**

```python
# src/risk/__init__.py
"""Risk management module for Phase 5."""

from .models import RiskCheckResult, DailyRiskState

__all__ = [
    "RiskCheckResult",
    "DailyRiskState",
]
```

```python
# src/risk/models.py
"""Data models for risk management."""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class RiskCheckResult:
    """Result of validating a trade against risk rules.

    Attributes:
        approved: Whether the trade can proceed.
        adjusted_quantity: Final quantity (0 if blocked).
        adjusted_value: Final dollar value of the position.
        rejection_reason: Why rejected (None if approved).
        warnings: Non-blocking warnings (e.g., unrealized drawdown).
    """

    approved: bool
    adjusted_quantity: int
    adjusted_value: float
    rejection_reason: str | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DailyRiskState:
    """Tracks running totals for the trading day.

    Attributes:
        date: The trading date.
        realized_pnl: Sum of closed position P&L.
        unrealized_pnl: Current open position P&L.
        trades_today: Number of trades executed today.
        is_blocked: True if daily loss limit has been hit.
    """

    date: date
    realized_pnl: float
    unrealized_pnl: float
    trades_today: int
    is_blocked: bool
```

```python
# tests/risk/__init__.py
"""Tests for risk management module."""
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/risk/test_models.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk
git add src/risk/__init__.py src/risk/models.py tests/risk/__init__.py tests/risk/test_models.py
git commit -m "feat(risk): add data models for risk management"
```

---

## Task 2: RiskManager - Core Structure and check_trade

**Files:**
- Create: `src/risk/risk_manager.py`
- Create: `tests/risk/test_risk_manager.py`
- Modify: `src/risk/__init__.py`

**Step 1: Write the failing tests**

```python
# tests/risk/test_risk_manager.py
"""Tests for RiskManager class."""

from datetime import date, datetime, timezone

import pytest

from src.risk.models import DailyRiskState, RiskCheckResult
from src.risk.risk_manager import RiskManager
from src.scoring.models import (
    Direction,
    ScoreComponents,
    ScoreTier,
    TradeRecommendation,
)


class TestRiskManagerInit:
    """Test RiskManager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        manager = RiskManager(
            max_position_value=1000.0,
            max_daily_loss=500.0,
        )
        assert manager._max_position_value == 1000.0
        assert manager._max_daily_loss == 500.0
        assert manager._unrealized_warning_threshold == 300.0

    def test_init_with_custom_warning_threshold(self):
        """Test initialization with custom warning threshold."""
        manager = RiskManager(
            max_position_value=2000.0,
            max_daily_loss=1000.0,
            unrealized_warning_threshold=500.0,
        )
        assert manager._unrealized_warning_threshold == 500.0


class TestRiskManagerCheckTrade:
    """Test check_trade method."""

    @pytest.fixture
    def manager(self) -> RiskManager:
        """Create a RiskManager instance for testing."""
        return RiskManager(
            max_position_value=1000.0,
            max_daily_loss=500.0,
            unrealized_warning_threshold=300.0,
        )

    @pytest.fixture
    def sample_recommendation(self) -> TradeRecommendation:
        """Create a sample trade recommendation."""
        return TradeRecommendation(
            symbol="NVDA",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=100.0,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=106.0,
            risk_reward_ratio=3.0,
            components=ScoreComponents(
                sentiment_score=90.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.6,
                confluence_bonus=0.1,
                credibility_multiplier=1.2,
                time_factor=1.0,
            ),
            reasoning="Strong bullish signal",
            timestamp=datetime.now(timezone.utc),
        )

    def test_approve_trade_within_limits(
        self, manager: RiskManager, sample_recommendation: TradeRecommendation
    ):
        """Trade should be approved when within all limits."""
        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=10,
            current_price=100.0,
        )
        assert result.approved is True
        assert result.adjusted_quantity == 10
        assert result.adjusted_value == 1000.0
        assert result.rejection_reason is None

    def test_reject_trade_exceeds_position_limit(
        self, manager: RiskManager, sample_recommendation: TradeRecommendation
    ):
        """Trade should be rejected when position value exceeds max."""
        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=20,  # 20 * $100 = $2000 > $1000 limit
            current_price=100.0,
        )
        assert result.approved is False
        assert result.adjusted_quantity == 0
        assert "position value" in result.rejection_reason.lower()

    def test_reject_trade_when_daily_blocked(
        self, manager: RiskManager, sample_recommendation: TradeRecommendation
    ):
        """Trade should be rejected when daily limit already hit."""
        # Simulate hitting daily limit
        manager._daily_state = DailyRiskState(
            date=date.today(),
            realized_pnl=-500.0,
            unrealized_pnl=0.0,
            trades_today=5,
            is_blocked=True,
        )

        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert result.approved is False
        assert result.adjusted_quantity == 0
        assert "daily loss limit" in result.rejection_reason.lower()

    def test_warn_on_unrealized_drawdown(
        self, manager: RiskManager, sample_recommendation: TradeRecommendation
    ):
        """Trade should be approved with warning when unrealized loss is high."""
        manager._daily_state = DailyRiskState(
            date=date.today(),
            realized_pnl=-100.0,
            unrealized_pnl=-350.0,  # Above warning threshold of $300
            trades_today=3,
            is_blocked=False,
        )

        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert result.approved is True
        assert len(result.warnings) > 0
        assert any("unrealized" in w.lower() for w in result.warnings)

    def test_reject_neutral_direction(
        self, manager: RiskManager
    ):
        """Trade should be rejected for NEUTRAL direction."""
        recommendation = TradeRecommendation(
            symbol="NVDA",
            direction=Direction.NEUTRAL,
            score=50.0,
            tier=ScoreTier.NO_TRADE,
            position_size_percent=0.0,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=102.0,
            risk_reward_ratio=2.0,
            components=ScoreComponents(
                sentiment_score=50.0,
                technical_score=50.0,
                sentiment_weight=0.5,
                technical_weight=0.5,
                confluence_bonus=0.0,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="Neutral signal",
            timestamp=datetime.now(timezone.utc),
        )

        result = manager.check_trade(
            recommendation=recommendation,
            requested_quantity=10,
            current_price=100.0,
        )
        assert result.approved is False
        assert "neutral" in result.rejection_reason.lower()
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/risk/test_risk_manager.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.risk.risk_manager'"

**Step 3: Write minimal implementation**

```python
# src/risk/risk_manager.py
"""Risk manager for validating trades against risk rules."""

from datetime import date

from src.risk.models import DailyRiskState, RiskCheckResult
from src.scoring.models import Direction, TradeRecommendation


class RiskManager:
    """Validates trades against risk rules.

    Enforces two simple rules:
    1. Max position value - No trade exceeds configured dollar amount
    2. Max daily loss - Stop trading after configured realized losses

    Attributes:
        _max_position_value: Maximum dollar value per trade.
        _max_daily_loss: Maximum realized loss before blocking.
        _unrealized_warning_threshold: Unrealized loss level that triggers warning.
        _daily_state: Current day's risk tracking state.
    """

    def __init__(
        self,
        max_position_value: float,
        max_daily_loss: float,
        unrealized_warning_threshold: float = 300.0,
    ):
        """Initialize the RiskManager.

        Args:
            max_position_value: Maximum dollar value per trade.
            max_daily_loss: Maximum realized loss per day before blocking.
            unrealized_warning_threshold: Unrealized loss that triggers warning.
        """
        self._max_position_value = max_position_value
        self._max_daily_loss = max_daily_loss
        self._unrealized_warning_threshold = unrealized_warning_threshold
        self._daily_state = DailyRiskState(
            date=date.today(),
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            trades_today=0,
            is_blocked=False,
        )

    def check_trade(
        self,
        recommendation: TradeRecommendation,
        requested_quantity: int,
        current_price: float,
    ) -> RiskCheckResult:
        """Validate a trade against risk rules.

        Checks in order:
        1. Is direction NEUTRAL? → Block
        2. Is daily limit already hit? → Block
        3. Does position value exceed max? → Block
        4. Is unrealized drawdown high? → Warn (don't block)

        Args:
            recommendation: The trade recommendation to validate.
            requested_quantity: Number of shares/contracts requested.
            current_price: Current price of the asset.

        Returns:
            RiskCheckResult indicating if trade is approved.
        """
        warnings: list[str] = []
        position_value = requested_quantity * current_price

        # Check 1: Reject NEUTRAL direction
        if recommendation.direction == Direction.NEUTRAL:
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0,
                adjusted_value=0.0,
                rejection_reason="Cannot trade NEUTRAL direction",
                warnings=[],
            )

        # Check 2: Is daily limit already hit?
        if self._daily_state.is_blocked:
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0,
                adjusted_value=0.0,
                rejection_reason="Daily loss limit already hit - trading blocked",
                warnings=[],
            )

        # Check 3: Does position value exceed max?
        if position_value > self._max_position_value:
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0,
                adjusted_value=0.0,
                rejection_reason=f"Position value ${position_value:.2f} exceeds max ${self._max_position_value:.2f}",
                warnings=[],
            )

        # Check 4: Unrealized drawdown warning (non-blocking)
        if abs(self._daily_state.unrealized_pnl) >= self._unrealized_warning_threshold:
            warnings.append(
                f"Unrealized loss at ${abs(self._daily_state.unrealized_pnl):.2f}, "
                f"threshold is ${self._unrealized_warning_threshold:.2f}"
            )

        return RiskCheckResult(
            approved=True,
            adjusted_quantity=requested_quantity,
            adjusted_value=position_value,
            rejection_reason=None,
            warnings=warnings,
        )
```

Update `src/risk/__init__.py`:

```python
# src/risk/__init__.py
"""Risk management module for Phase 5."""

from .models import RiskCheckResult, DailyRiskState
from .risk_manager import RiskManager

__all__ = [
    "RiskCheckResult",
    "DailyRiskState",
    "RiskManager",
]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/risk/test_risk_manager.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk
git add src/risk/risk_manager.py src/risk/__init__.py tests/risk/test_risk_manager.py
git commit -m "feat(risk): add RiskManager with check_trade method"
```

---

## Task 3: RiskManager - Recording and State Management

**Files:**
- Modify: `src/risk/risk_manager.py`
- Modify: `tests/risk/test_risk_manager.py`

**Step 1: Add failing tests for record_trade, record_close, update_unrealized_pnl, reset**

```python
# Add to tests/risk/test_risk_manager.py

class TestRiskManagerRecording:
    """Test recording methods."""

    @pytest.fixture
    def manager(self) -> RiskManager:
        """Create a RiskManager instance for testing."""
        return RiskManager(
            max_position_value=1000.0,
            max_daily_loss=500.0,
        )

    def test_record_trade_increments_count(self, manager: RiskManager):
        """Recording a trade should increment trades_today."""
        assert manager._daily_state.trades_today == 0
        manager.record_trade(symbol="NVDA", quantity=10, price=100.0)
        assert manager._daily_state.trades_today == 1

    def test_record_close_updates_realized_pnl(self, manager: RiskManager):
        """Recording a close should update realized P&L."""
        manager.record_close(symbol="NVDA", pnl=-150.0)
        assert manager._daily_state.realized_pnl == -150.0

    def test_record_close_blocks_when_limit_hit(self, manager: RiskManager):
        """Should block trading when realized loss exceeds limit."""
        manager.record_close(symbol="NVDA", pnl=-300.0)
        assert manager._daily_state.is_blocked is False

        manager.record_close(symbol="AAPL", pnl=-250.0)  # Total: -550
        assert manager._daily_state.realized_pnl == -550.0
        assert manager._daily_state.is_blocked is True

    def test_update_unrealized_pnl(self, manager: RiskManager):
        """Should update unrealized P&L."""
        manager.update_unrealized_pnl(-200.0)
        assert manager._daily_state.unrealized_pnl == -200.0

    def test_reset_daily_state(self, manager: RiskManager):
        """Reset should clear all daily state."""
        # Accumulate some state
        manager.record_trade(symbol="NVDA", quantity=10, price=100.0)
        manager.record_close(symbol="NVDA", pnl=-100.0)
        manager.update_unrealized_pnl(-50.0)

        # Reset
        manager.reset_daily_state()

        state = manager.get_daily_state()
        assert state.realized_pnl == 0.0
        assert state.unrealized_pnl == 0.0
        assert state.trades_today == 0
        assert state.is_blocked is False
        assert state.date == date.today()

    def test_get_daily_state(self, manager: RiskManager):
        """Should return current daily state."""
        state = manager.get_daily_state()
        assert isinstance(state, DailyRiskState)
        assert state.date == date.today()
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/risk/test_risk_manager.py::TestRiskManagerRecording -v`
Expected: FAIL with "AttributeError: 'RiskManager' object has no attribute 'record_trade'"

**Step 3: Add implementation**

Add these methods to `src/risk/risk_manager.py` in the RiskManager class:

```python
    def record_trade(self, symbol: str, quantity: int, price: float) -> None:
        """Record an executed trade.

        Args:
            symbol: Stock ticker symbol.
            quantity: Number of shares traded.
            price: Price per share.
        """
        self._daily_state.trades_today += 1

    def record_close(self, symbol: str, pnl: float) -> None:
        """Record a closed position and its realized P&L.

        If realized P&L drops below -max_daily_loss, sets is_blocked=True.

        Args:
            symbol: Stock ticker symbol.
            pnl: Realized profit/loss from the closed position.
        """
        self._daily_state.realized_pnl += pnl

        # Check if daily loss limit is hit
        if self._daily_state.realized_pnl <= -self._max_daily_loss:
            self._daily_state.is_blocked = True

    def update_unrealized_pnl(self, total_unrealized: float) -> None:
        """Update current unrealized P&L.

        Called periodically to track open position values.

        Args:
            total_unrealized: Total unrealized P&L across all open positions.
        """
        self._daily_state.unrealized_pnl = total_unrealized

    def reset_daily_state(self) -> None:
        """Reset for new trading day.

        Should be called at market open each day.
        """
        self._daily_state = DailyRiskState(
            date=date.today(),
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            trades_today=0,
            is_blocked=False,
        )

    def get_daily_state(self) -> DailyRiskState:
        """Get current daily risk state for monitoring.

        Returns:
            Current DailyRiskState.
        """
        return self._daily_state
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/risk/test_risk_manager.py -v`
Expected: PASS (14 tests)

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk
git add src/risk/risk_manager.py tests/risk/test_risk_manager.py
git commit -m "feat(risk): add recording and state management methods"
```

---

## Task 4: RiskSettings Configuration

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`
- Create: `tests/config/test_risk_settings.py`

**Step 1: Write the failing tests**

```python
# tests/config/test_risk_settings.py
"""Tests for RiskSettings configuration."""

import pytest
from pydantic import ValidationError

from src.config.settings import RiskSettings


class TestRiskSettings:
    """Test suite for RiskSettings."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = RiskSettings()
        assert settings.enabled is True
        assert settings.max_position_value == 1000.0
        assert settings.max_daily_loss == 500.0
        assert settings.unrealized_warning_threshold == 300.0

    def test_custom_values(self):
        """Test custom values can be set."""
        settings = RiskSettings(
            enabled=False,
            max_position_value=2000.0,
            max_daily_loss=1000.0,
            unrealized_warning_threshold=500.0,
        )
        assert settings.enabled is False
        assert settings.max_position_value == 2000.0
        assert settings.max_daily_loss == 1000.0
        assert settings.unrealized_warning_threshold == 500.0

    def test_max_position_value_must_be_positive(self):
        """max_position_value must be > 0."""
        with pytest.raises(ValidationError):
            RiskSettings(max_position_value=0)

        with pytest.raises(ValidationError):
            RiskSettings(max_position_value=-100)

    def test_max_daily_loss_must_be_positive(self):
        """max_daily_loss must be > 0."""
        with pytest.raises(ValidationError):
            RiskSettings(max_daily_loss=0)

    def test_unrealized_warning_threshold_must_be_positive(self):
        """unrealized_warning_threshold must be > 0."""
        with pytest.raises(ValidationError):
            RiskSettings(unrealized_warning_threshold=-50)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/config/test_risk_settings.py -v`
Expected: FAIL with "ImportError: cannot import name 'RiskSettings'"

**Step 3: Write implementation**

Add to `src/config/settings.py` (after ScoringSettings class):

```python
class RiskSettings(BaseModel):
    """Settings for risk management."""

    enabled: bool = True
    max_position_value: float = Field(default=1000.0, gt=0)
    max_daily_loss: float = Field(default=500.0, gt=0)
    unrealized_warning_threshold: float = Field(default=300.0, gt=0)
```

Add `risk` field to the `Settings` class:

```python
    risk: RiskSettings = Field(default_factory=RiskSettings)
```

Update `config/settings.yaml` to add risk section:

```yaml
# Risk Management Settings
risk:
  enabled: true
  max_position_value: 1000.0
  max_daily_loss: 500.0
  unrealized_warning_threshold: 300.0
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/config/test_risk_settings.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk
git add src/config/settings.py config/settings.yaml tests/config/test_risk_settings.py
git commit -m "feat(config): add risk settings"
```

---

## Task 5: Integration Tests

**Files:**
- Create: `tests/integration/test_risk_pipeline.py`

**Step 1: Write the integration tests**

```python
# tests/integration/test_risk_pipeline.py
"""Integration tests for risk management pipeline."""

from datetime import datetime, timezone

import pytest

from src.risk.models import DailyRiskState
from src.risk.risk_manager import RiskManager
from src.scoring.models import (
    Direction,
    ScoreComponents,
    ScoreTier,
    TradeRecommendation,
)


class TestRiskPipelineIntegration:
    """Integration tests for scoring → risk pipeline."""

    @pytest.fixture
    def risk_manager(self) -> RiskManager:
        """Create a RiskManager for testing."""
        return RiskManager(
            max_position_value=1000.0,
            max_daily_loss=500.0,
            unrealized_warning_threshold=300.0,
        )

    @pytest.fixture
    def strong_recommendation(self) -> TradeRecommendation:
        """Create a strong LONG recommendation."""
        return TradeRecommendation(
            symbol="NVDA",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=100.0,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=106.0,
            risk_reward_ratio=3.0,
            components=ScoreComponents(
                sentiment_score=90.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.6,
                confluence_bonus=0.1,
                credibility_multiplier=1.2,
                time_factor=1.0,
            ),
            reasoning="Strong bullish signal",
            timestamp=datetime.now(timezone.utc),
        )

    def test_full_trading_day_simulation(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ):
        """Simulate a full trading day with multiple trades."""
        # Trade 1: Approved, within limits
        result1 = risk_manager.check_trade(strong_recommendation, 10, 100.0)
        assert result1.approved is True
        risk_manager.record_trade("NVDA", 10, 100.0)

        # Close Trade 1 with small loss
        risk_manager.record_close("NVDA", -100.0)
        assert risk_manager.get_daily_state().realized_pnl == -100.0

        # Trade 2: Approved
        result2 = risk_manager.check_trade(strong_recommendation, 8, 100.0)
        assert result2.approved is True
        risk_manager.record_trade("NVDA", 8, 100.0)

        # Close Trade 2 with larger loss
        risk_manager.record_close("NVDA", -300.0)
        assert risk_manager.get_daily_state().realized_pnl == -400.0
        assert risk_manager.get_daily_state().is_blocked is False

        # Trade 3: Approved (still under limit)
        result3 = risk_manager.check_trade(strong_recommendation, 5, 100.0)
        assert result3.approved is True
        risk_manager.record_trade("NVDA", 5, 100.0)

        # Close Trade 3 - this triggers the block
        risk_manager.record_close("NVDA", -150.0)
        assert risk_manager.get_daily_state().realized_pnl == -550.0
        assert risk_manager.get_daily_state().is_blocked is True

        # Trade 4: Blocked
        result4 = risk_manager.check_trade(strong_recommendation, 5, 100.0)
        assert result4.approved is False
        assert "daily loss limit" in result4.rejection_reason.lower()

    def test_unrealized_warning_during_trading(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ):
        """Test unrealized warning is triggered during trading."""
        # Execute a trade
        result1 = risk_manager.check_trade(strong_recommendation, 10, 100.0)
        assert result1.approved is True
        assert len(result1.warnings) == 0

        # Update unrealized P&L to trigger warning
        risk_manager.update_unrealized_pnl(-350.0)

        # Next trade check should have warning
        result2 = risk_manager.check_trade(strong_recommendation, 5, 100.0)
        assert result2.approved is True  # Still approved
        assert len(result2.warnings) > 0
        assert any("unrealized" in w.lower() for w in result2.warnings)

    def test_position_size_limit_enforcement(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ):
        """Test position size limit is enforced."""
        # Try to trade $1500 worth (exceeds $1000 limit)
        result = risk_manager.check_trade(strong_recommendation, 15, 100.0)
        assert result.approved is False
        assert "position value" in result.rejection_reason.lower()
        assert "$1500" in result.rejection_reason

    def test_daily_reset_allows_trading_again(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ):
        """Test that daily reset allows trading after being blocked."""
        # Hit the daily limit
        risk_manager.record_close("NVDA", -600.0)
        assert risk_manager.get_daily_state().is_blocked is True

        # Verify blocked
        result1 = risk_manager.check_trade(strong_recommendation, 5, 100.0)
        assert result1.approved is False

        # Reset for new day
        risk_manager.reset_daily_state()

        # Should be able to trade again
        result2 = risk_manager.check_trade(strong_recommendation, 5, 100.0)
        assert result2.approved is True

    def test_short_direction_allowed(self, risk_manager: RiskManager):
        """Test SHORT direction trades are allowed."""
        short_rec = TradeRecommendation(
            symbol="TSLA",
            direction=Direction.SHORT,
            score=75.0,
            tier=ScoreTier.MODERATE,
            position_size_percent=50.0,
            entry_price=200.0,
            stop_loss=204.0,
            take_profit=188.0,
            risk_reward_ratio=3.0,
            components=ScoreComponents(
                sentiment_score=25.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.6,
                confluence_bonus=0.0,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="Bearish signal",
            timestamp=datetime.now(timezone.utc),
        )

        result = risk_manager.check_trade(short_rec, 5, 200.0)
        assert result.approved is True
        assert result.adjusted_value == 1000.0

    def test_trades_count_accumulates(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ):
        """Test that trades_today counter accumulates."""
        assert risk_manager.get_daily_state().trades_today == 0

        for i in range(5):
            result = risk_manager.check_trade(strong_recommendation, 5, 100.0)
            if result.approved:
                risk_manager.record_trade("NVDA", 5, 100.0)

        assert risk_manager.get_daily_state().trades_today == 5
```

**Step 2: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest tests/integration/test_risk_pipeline.py -v`
Expected: PASS (7 tests)

**Step 3: Commit**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk
git add tests/integration/test_risk_pipeline.py
git commit -m "test: add integration tests for risk pipeline"
```

---

## Task 6: Final Test Suite Run

**Step 1: Run full test suite**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase5-risk && uv run pytest --cov=src --cov-report=term-missing -v`

Expected: All tests pass, coverage >90% on risk module

**Step 2: Verify coverage**

Check that `src/risk/` has >90% coverage.

**Step 3: Commit any fixes if needed**

If any tests fail, fix them and commit.

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Data Models | 6 |
| 2 | RiskManager check_trade | 8 |
| 3 | Recording & State | 6 |
| 4 | RiskSettings | 5 |
| 5 | Integration Tests | 7 |
| 6 | Final Verification | - |

**Total new tests:** ~32
