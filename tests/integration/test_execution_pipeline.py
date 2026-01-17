"""Integration tests for execution pipeline."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.execution.trade_executor import TradeExecutor
from src.risk.risk_manager import RiskManager
from src.scoring.models import (
    Direction,
    ScoreComponents,
    ScoreTier,
    TradeRecommendation,
)


def make_recommendation(
    symbol: str = "AAPL",
    direction: Direction = Direction.LONG,
    entry_price: float = 150.00,
    stop_loss: float = 147.00,
    take_profit: float = 156.00,
) -> TradeRecommendation:
    """Helper to create TradeRecommendation."""
    return TradeRecommendation(
        symbol=symbol,
        direction=direction,
        score=85.0,
        tier=ScoreTier.STRONG,
        position_size_percent=100.0,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=2.0,
        components=ScoreComponents(
            sentiment_score=90.0,
            technical_score=80.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.0,
            credibility_multiplier=1.0,
            time_factor=1.0,
        ),
        reasoning="Test",
        timestamp=datetime.now(),
    )


class TestExecutionPipelineIntegration:
    """Integration tests for Risk → Execution pipeline."""

    @pytest.fixture
    def mock_alpaca(self):
        """Create mock AlpacaClient."""
        client = AsyncMock()
        client.get_all_positions = AsyncMock(return_value=[])
        client.submit_order = AsyncMock(
            return_value={
                "id": "order-123",
                "status": "filled",
                "symbol": "AAPL",
                "qty": 10,
                "side": "buy",
                "type": "market",
                "filled_qty": 10,
                "filled_avg_price": 150.00,
            }
        )
        client.get_order = AsyncMock(
            return_value={
                "filled_avg_price": 156.00,
                "status": "filled",
            }
        )
        return client

    @pytest.fixture
    def risk_manager(self):
        """Create real RiskManager."""
        return RiskManager(
            max_position_value=2000.0,
            max_daily_loss=500.0,
            unrealized_warning_threshold=300.0,
        )

    @pytest.fixture
    def executor(self, mock_alpaca, risk_manager):
        """Create TradeExecutor with real RiskManager."""
        return TradeExecutor(mock_alpaca, risk_manager)

    @pytest.mark.asyncio
    async def test_full_flow_risk_to_execution(self, executor, risk_manager):
        """Test full flow: recommendation → risk check → execution."""
        recommendation = make_recommendation()

        # Risk check
        risk_result = risk_manager.check_trade(
            recommendation=recommendation,
            requested_quantity=10,
            current_price=150.00,
        )
        assert risk_result.approved is True

        # Execute
        result = await executor.execute(recommendation, risk_result)

        assert result.success is True
        assert result.symbol == "AAPL"
        assert risk_manager.get_daily_state().trades_today == 1

    @pytest.mark.asyncio
    async def test_position_close_updates_risk_manager(
        self, executor, risk_manager, mock_alpaca
    ):
        """Test that closed position P&L flows to RiskManager."""
        recommendation = make_recommendation()
        risk_result = risk_manager.check_trade(recommendation, 10, 150.00)

        # Execute trade
        await executor.execute(recommendation, risk_result)
        assert "AAPL" in executor.get_tracked_positions()

        # Simulate position close (Alpaca returns empty)
        mock_alpaca.get_all_positions = AsyncMock(return_value=[])

        # Sync should detect close and record P&L
        closed = await executor.sync_positions()

        assert "AAPL" in closed
        # P&L = (156 - 150) * 10 = 60 (profit)
        assert risk_manager.get_daily_state().realized_pnl == 60.0

    @pytest.mark.asyncio
    async def test_daily_loss_blocks_after_losses(
        self, executor, risk_manager, mock_alpaca
    ):
        """Test that hitting daily loss limit blocks further trades."""
        # First trade
        rec1 = make_recommendation(symbol="AAPL")
        risk1 = risk_manager.check_trade(rec1, 10, 150.00)
        await executor.execute(rec1, risk1)

        # Simulate big loss (closed at $100 instead of $150)
        mock_alpaca.get_all_positions = AsyncMock(return_value=[])
        mock_alpaca.get_order = AsyncMock(
            return_value={
                "filled_avg_price": 100.00,
                "status": "filled",
            }
        )
        await executor.sync_positions()

        # P&L = (100 - 150) * 10 = -500 (equals max_daily_loss)
        assert risk_manager.get_daily_state().is_blocked is True

        # Next trade should be blocked
        rec2 = make_recommendation(symbol="TSLA")
        risk2 = risk_manager.check_trade(rec2, 5, 200.00)
        assert risk2.approved is False

    @pytest.mark.asyncio
    async def test_multiple_trades_accumulate(
        self, executor, risk_manager, mock_alpaca
    ):
        """Test multiple trades accumulate correctly."""
        # Trade 1
        rec1 = make_recommendation(symbol="AAPL")
        risk1 = risk_manager.check_trade(rec1, 10, 150.00)
        await executor.execute(rec1, risk1)

        # Update mock to show AAPL position still exists when we check before Trade 2
        mock_alpaca.get_all_positions = AsyncMock(
            return_value=[
                {"symbol": "AAPL", "qty": 10, "unrealized_pl": 0.0},
            ]
        )

        # Trade 2
        mock_alpaca.submit_order = AsyncMock(
            return_value={
                "id": "order-456",
                "status": "filled",
                "symbol": "TSLA",
                "qty": 5,
                "side": "buy",
                "type": "market",
                "filled_qty": 5,
                "filled_avg_price": 200.00,
            }
        )
        rec2 = make_recommendation(
            symbol="TSLA",
            entry_price=200.00,
            stop_loss=194.00,
            take_profit=212.00,
        )
        risk2 = risk_manager.check_trade(rec2, 5, 200.00)
        await executor.execute(rec2, risk2)

        assert risk_manager.get_daily_state().trades_today == 2
        assert len(executor.get_tracked_positions()) == 2
