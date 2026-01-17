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
    ) -> None:
        """Simulate full day: trades, losses, eventually blocked.

        - Trade 1: Approved, execute, close with -100 loss
        - Trade 2: Approved, execute, close with -300 loss (total -400)
        - Trade 3: Approved, execute, close with -150 loss (total -550, triggers block)
        - Trade 4: REJECTED - daily limit hit
        """
        # Trade 1: Approved, execute, close with -100 loss
        approval1 = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert approval1.approved
        assert len(approval1.warnings) == 0

        risk_manager.record_trade(
            symbol="NVDA",
            quantity=5,
            price=100.0,
        )
        risk_manager.record_close(symbol="NVDA", pnl=-100.0)

        state = risk_manager.get_daily_state()
        assert state.realized_pnl == -100.0
        assert state.trades_today == 1

        # Trade 2: Approved, execute, close with -300 loss (total -400)
        approval2 = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert approval2.approved

        risk_manager.record_trade(
            symbol="NVDA",
            quantity=5,
            price=100.0,
        )
        risk_manager.record_close(symbol="NVDA", pnl=-300.0)

        state = risk_manager.get_daily_state()
        assert state.realized_pnl == -400.0
        assert state.trades_today == 2

        # Trade 3: Approved, execute, close with -150 loss (total -550, triggers block)
        approval3 = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert approval3.approved

        risk_manager.record_trade(
            symbol="NVDA",
            quantity=5,
            price=100.0,
        )
        risk_manager.record_close(symbol="NVDA", pnl=-150.0)

        state = risk_manager.get_daily_state()
        assert state.realized_pnl == -550.0
        assert state.trades_today == 3
        assert state.is_blocked

        # Trade 4: REJECTED - daily limit hit
        approval4 = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert not approval4.approved
        assert "daily loss limit" in approval4.rejection_reason.lower()

    def test_unrealized_warning_during_trading(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ) -> None:
        """Test warning when unrealized loss exceeds threshold.

        - Trade 1: Approved, no warnings
        - Update unrealized to -350
        - Trade 2: Approved WITH warning about unrealized
        """
        # Trade 1: Approved, no warnings
        approval1 = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert approval1.approved
        assert len(approval1.warnings) == 0

        risk_manager.record_trade(
            symbol="NVDA",
            quantity=5,
            price=100.0,
        )

        # Update unrealized to -350
        risk_manager.update_unrealized_pnl(total_unrealized=-350.0)
        state = risk_manager.get_daily_state()
        assert state.unrealized_pnl == -350.0

        # Trade 2: Approved WITH warning about unrealized
        approval2 = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert approval2.approved
        assert len(approval2.warnings) > 0
        assert "drawdown" in approval2.warnings[0].lower()

    def test_position_size_limit_enforcement(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ) -> None:
        """Test position value limit.

        - Request 15 shares at $100 = $1500 > $1000 limit
        - Should be rejected
        """
        approval = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=15,  # 15 * $100 = $1500 > $1000 limit
            current_price=100.0,
        )
        assert not approval.approved
        assert "position value" in approval.rejection_reason.lower()

    def test_daily_reset_allows_trading_again(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ) -> None:
        """Test reset unblocks trading.

        - Hit daily limit, verify blocked
        - Call reset_daily_state()
        - Verify can trade again
        """
        # Hit daily limit by simulating large loss
        risk_manager.record_trade(
            symbol="NVDA",
            quantity=5,
            price=100.0,
        )
        risk_manager.record_close(symbol="NVDA", pnl=-600.0)

        state = risk_manager.get_daily_state()
        assert state.is_blocked

        # Verify blocked
        approval_blocked = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert not approval_blocked.approved

        # Reset daily state
        risk_manager.reset_daily_state()

        state = risk_manager.get_daily_state()
        assert state.realized_pnl == 0.0
        assert state.unrealized_pnl == 0.0
        assert state.trades_today == 0
        assert not state.is_blocked

        # Verify can trade again
        approval_after_reset = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert approval_after_reset.approved

    def test_short_direction_allowed(self, risk_manager: RiskManager) -> None:
        """Test SHORT direction trades work.

        - Create bearish recommendation with Direction.SHORT
        - Verify approved
        """
        short_recommendation = TradeRecommendation(
            symbol="TSLA",
            direction=Direction.SHORT,
            score=75.0,
            tier=ScoreTier.STRONG,
            position_size_percent=80.0,
            entry_price=200.0,
            stop_loss=205.0,
            take_profit=190.0,
            risk_reward_ratio=2.0,
            components=ScoreComponents(
                sentiment_score=70.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.6,
                confluence_bonus=0.05,
                credibility_multiplier=1.1,
                time_factor=1.0,
            ),
            reasoning="Strong bearish signal",
            timestamp=datetime.now(timezone.utc),
        )

        approval = risk_manager.check_trade(
            recommendation=short_recommendation,
            requested_quantity=4,  # 4 * $200 = $800 < $1000 limit
            current_price=200.0,
        )
        assert approval.approved
        assert len(approval.warnings) == 0

    def test_trades_count_accumulates(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ) -> None:
        """Test trades_today increments correctly."""
        state = risk_manager.get_daily_state()
        assert state.trades_today == 0

        # Execute first trade
        risk_manager.record_trade(
            symbol="NVDA",
            quantity=5,
            price=100.0,
        )
        state = risk_manager.get_daily_state()
        assert state.trades_today == 1

        # Close trade
        risk_manager.record_close(symbol="NVDA", pnl=50.0)
        state = risk_manager.get_daily_state()
        assert state.trades_today == 1  # Count doesn't decrease

        # Execute second trade
        risk_manager.record_trade(
            symbol="NVDA",
            quantity=5,
            price=100.0,
        )
        state = risk_manager.get_daily_state()
        assert state.trades_today == 2

        # Execute third trade
        risk_manager.record_trade(
            symbol="AAPL",
            quantity=3,
            price=150.0,
        )
        state = risk_manager.get_daily_state()
        assert state.trades_today == 3

    def test_multiple_symbols_same_session(
        self, risk_manager: RiskManager, strong_recommendation: TradeRecommendation
    ) -> None:
        """Test trading multiple symbols in one day.

        - Trade NVDA, close with small profit
        - Trade AAPL, close with loss
        - Verify cumulative P&L tracking works
        """
        # Trade NVDA with profit
        approval_nvda = risk_manager.check_trade(
            recommendation=strong_recommendation,
            requested_quantity=5,
            current_price=100.0,
        )
        assert approval_nvda.approved

        risk_manager.record_trade(
            symbol="NVDA",
            quantity=5,
            price=100.0,
        )
        risk_manager.record_close(symbol="NVDA", pnl=75.0)

        state = risk_manager.get_daily_state()
        assert state.realized_pnl == 75.0
        assert state.trades_today == 1

        # Trade AAPL with loss
        aapl_recommendation = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=80.0,
            tier=ScoreTier.STRONG,
            position_size_percent=100.0,
            entry_price=150.0,
            stop_loss=148.0,
            take_profit=156.0,
            risk_reward_ratio=3.0,
            components=ScoreComponents(
                sentiment_score=85.0,
                technical_score=75.0,
                sentiment_weight=0.4,
                technical_weight=0.6,
                confluence_bonus=0.08,
                credibility_multiplier=1.15,
                time_factor=1.0,
            ),
            reasoning="Strong bullish momentum",
            timestamp=datetime.now(timezone.utc),
        )

        approval_aapl = risk_manager.check_trade(
            recommendation=aapl_recommendation,
            requested_quantity=4,  # 4 * $150 = $600 < $1000 limit
            current_price=150.0,
        )
        assert approval_aapl.approved

        risk_manager.record_trade(
            symbol="AAPL",
            quantity=4,
            price=150.0,
        )
        risk_manager.record_close(symbol="AAPL", pnl=-125.0)

        # Verify cumulative tracking
        state = risk_manager.get_daily_state()
        assert state.realized_pnl == -50.0  # 75 - 125
        assert state.trades_today == 2
        assert not state.is_blocked
