"""Tests for RiskManager class."""

from datetime import datetime, date

import pytest

from src.risk.models import DailyRiskState
from src.risk.risk_manager import RiskManager
from src.scoring.models import (
    Direction,
    ScoreComponents,
    ScoreTier,
    TradeRecommendation,
)


class TestRiskManagerInit:
    """Tests for RiskManager initialization."""

    def test_init_with_defaults(self):
        """Test RiskManager initialization with default warning threshold."""
        manager = RiskManager(
            max_position_value=1000.0,
            max_daily_loss=500.0,
        )

        assert manager.max_position_value == 1000.0
        assert manager.max_daily_loss == 500.0
        assert manager.unrealized_warning_threshold == 300.0
        assert isinstance(manager._daily_state, DailyRiskState)
        assert manager._daily_state.date == date.today()
        assert manager._daily_state.realized_pnl == 0.0
        assert manager._daily_state.unrealized_pnl == 0.0
        assert manager._daily_state.trades_today == 0
        assert manager._daily_state.is_blocked is False

    def test_init_with_custom_warning_threshold(self):
        """Test RiskManager initialization with custom warning threshold."""
        manager = RiskManager(
            max_position_value=2000.0,
            max_daily_loss=1000.0,
            unrealized_warning_threshold=500.0,
        )

        assert manager.max_position_value == 2000.0
        assert manager.max_daily_loss == 1000.0
        assert manager.unrealized_warning_threshold == 500.0
        assert isinstance(manager._daily_state, DailyRiskState)


class TestRiskManagerCheckTrade:
    """Tests for RiskManager.check_trade method."""

    @pytest.fixture
    def manager(self):
        """Create a RiskManager instance for testing."""
        return RiskManager(
            max_position_value=1000.0,
            max_daily_loss=500.0,
            unrealized_warning_threshold=300.0,
        )

    @pytest.fixture
    def sample_recommendation(self):
        """Create a sample TradeRecommendation for testing."""
        return TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=75.0,
            tier=ScoreTier.MODERATE,
            position_size_percent=50.0,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            risk_reward_ratio=2.0,
            components=ScoreComponents(
                sentiment_score=70.0,
                technical_score=80.0,
                sentiment_weight=0.5,
                technical_weight=0.5,
                confluence_bonus=0.05,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="Strong technical setup with positive sentiment.",
            timestamp=datetime.now(),
        )

    def test_approve_trade_within_limits(self, manager, sample_recommendation):
        """Test that a trade within all limits is approved."""
        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=5,
            current_price=150.0,
        )

        assert result.approved is True
        assert result.adjusted_quantity == 5
        assert result.adjusted_value == 750.0  # 5 * 150.0
        assert result.rejection_reason is None
        assert len(result.warnings) == 0

    def test_reject_trade_exceeds_position_limit(self, manager, sample_recommendation):
        """Test that a trade exceeding max position value is rejected."""
        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=10,  # 10 * 150.0 = 1500.0 > 1000.0
            current_price=150.0,
        )

        assert result.approved is False
        assert result.adjusted_quantity == 0
        assert result.adjusted_value == 0.0
        assert result.rejection_reason == "Position value ($1500.00) exceeds maximum allowed ($1000.00)"
        assert len(result.warnings) == 0

    def test_reject_trade_when_daily_blocked(self, manager, sample_recommendation):
        """Test that trades are rejected when daily trading is blocked."""
        # Manually set the daily state to blocked
        manager._daily_state.is_blocked = True

        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=5,
            current_price=150.0,
        )

        assert result.approved is False
        assert result.adjusted_quantity == 0
        assert result.adjusted_value == 0.0
        assert result.rejection_reason == "Trading blocked: daily loss limit exceeded"
        assert len(result.warnings) == 0

    def test_warn_on_unrealized_drawdown(self, manager, sample_recommendation):
        """Test that a warning is added when unrealized PnL exceeds threshold."""
        # Set unrealized PnL to trigger warning
        manager._daily_state.unrealized_pnl = -350.0  # Exceeds 300.0 threshold

        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=5,
            current_price=150.0,
        )

        assert result.approved is True  # Trade still approved
        assert result.adjusted_quantity == 5
        assert result.adjusted_value == 750.0
        assert result.rejection_reason is None
        assert len(result.warnings) == 1
        assert "Unrealized drawdown" in result.warnings[0]
        assert "$350.00" in result.warnings[0]

    def test_reject_neutral_direction(self, manager, sample_recommendation):
        """Test that trades with NEUTRAL direction are rejected."""
        # Modify recommendation to have NEUTRAL direction
        sample_recommendation.direction = Direction.NEUTRAL

        result = manager.check_trade(
            recommendation=sample_recommendation,
            requested_quantity=5,
            current_price=150.0,
        )

        assert result.approved is False
        assert result.adjusted_quantity == 0
        assert result.adjusted_value == 0.0
        assert result.rejection_reason == "Cannot execute trade with NEUTRAL direction"
        assert len(result.warnings) == 0


class TestRiskManagerRecording:
    """Tests for RiskManager recording and state management methods."""

    @pytest.fixture
    def manager(self):
        """Create a RiskManager instance for testing."""
        return RiskManager(
            max_position_value=1000.0,
            max_daily_loss=500.0,
            unrealized_warning_threshold=300.0,
        )

    def test_record_trade_increments_count(self, manager):
        """Test that record_trade increments the trades_today counter."""
        assert manager._daily_state.trades_today == 0

        manager.record_trade(symbol="AAPL", quantity=10, price=150.0)
        assert manager._daily_state.trades_today == 1

        manager.record_trade(symbol="MSFT", quantity=5, price=300.0)
        assert manager._daily_state.trades_today == 2

        manager.record_trade(symbol="AAPL", quantity=3, price=152.0)
        assert manager._daily_state.trades_today == 3

    def test_record_close_updates_realized_pnl(self, manager):
        """Test that record_close updates realized PnL correctly."""
        assert manager._daily_state.realized_pnl == 0.0
        assert manager._daily_state.is_blocked is False

        # Record a profit
        manager.record_close(symbol="AAPL", pnl=100.0)
        assert manager._daily_state.realized_pnl == 100.0
        assert manager._daily_state.is_blocked is False

        # Record a small loss
        manager.record_close(symbol="MSFT", pnl=-50.0)
        assert manager._daily_state.realized_pnl == 50.0
        assert manager._daily_state.is_blocked is False

    def test_record_close_blocks_when_limit_hit(self, manager):
        """Test that record_close blocks trading when daily loss limit is hit."""
        assert manager._daily_state.is_blocked is False

        # First close with loss of 300 (not yet at limit of 500)
        manager.record_close(symbol="AAPL", pnl=-300.0)
        assert manager._daily_state.realized_pnl == -300.0
        assert manager._daily_state.is_blocked is False

        # Second close with loss of 250, total = -550 (exceeds limit of 500)
        manager.record_close(symbol="MSFT", pnl=-250.0)
        assert manager._daily_state.realized_pnl == -550.0
        assert manager._daily_state.is_blocked is True

    def test_update_unrealized_pnl(self, manager):
        """Test that update_unrealized_pnl sets the unrealized PnL value."""
        assert manager._daily_state.unrealized_pnl == 0.0

        manager.update_unrealized_pnl(total_unrealized=150.0)
        assert manager._daily_state.unrealized_pnl == 150.0

        manager.update_unrealized_pnl(total_unrealized=-75.0)
        assert manager._daily_state.unrealized_pnl == -75.0

        manager.update_unrealized_pnl(total_unrealized=0.0)
        assert manager._daily_state.unrealized_pnl == 0.0

    def test_reset_daily_state(self, manager):
        """Test that reset_daily_state clears all accumulated state."""
        # Accumulate some state
        manager.record_trade(symbol="AAPL", quantity=10, price=150.0)
        manager.record_trade(symbol="MSFT", quantity=5, price=300.0)
        manager.record_close(symbol="AAPL", pnl=-200.0)
        manager.update_unrealized_pnl(total_unrealized=-100.0)

        # Verify state is accumulated
        assert manager._daily_state.trades_today == 2
        assert manager._daily_state.realized_pnl == -200.0
        assert manager._daily_state.unrealized_pnl == -100.0

        # Reset the state
        manager.reset_daily_state()

        # Verify everything is cleared
        assert manager._daily_state.date == date.today()
        assert manager._daily_state.realized_pnl == 0.0
        assert manager._daily_state.unrealized_pnl == 0.0
        assert manager._daily_state.trades_today == 0
        assert manager._daily_state.is_blocked is False

    def test_get_daily_state(self, manager):
        """Test that get_daily_state returns the current daily risk state."""
        # Get initial state
        state = manager.get_daily_state()
        assert isinstance(state, DailyRiskState)
        assert state.date == date.today()
        assert state.realized_pnl == 0.0
        assert state.unrealized_pnl == 0.0
        assert state.trades_today == 0
        assert state.is_blocked is False

        # Accumulate some state
        manager.record_trade(symbol="AAPL", quantity=10, price=150.0)
        manager.record_close(symbol="AAPL", pnl=75.0)
        manager.update_unrealized_pnl(total_unrealized=-25.0)

        # Get updated state
        state = manager.get_daily_state()
        assert state.trades_today == 1
        assert state.realized_pnl == 75.0
        assert state.unrealized_pnl == -25.0
        assert state.is_blocked is False
