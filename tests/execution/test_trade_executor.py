# tests/execution/test_trade_executor.py
"""Tests for TradeExecutor class."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock

from src.execution.trade_executor import TradeExecutor
from src.execution.alpaca_client import AlpacaClient
from src.execution.models import ExecutionResult, TrackedPosition
from src.risk.risk_manager import RiskManager
from src.scoring.models import Direction, TradeRecommendation, ScoreComponents, ScoreTier
from src.risk.models import RiskCheckResult


# Helper functions for test fixtures
def make_recommendation(
    symbol: str = "AAPL",
    direction: Direction = Direction.LONG,
    entry_price: float = 150.00,
    stop_loss: float = 147.00,
    take_profit: float = 156.00,
) -> TradeRecommendation:
    """Create a test TradeRecommendation with sensible defaults."""
    return TradeRecommendation(
        symbol=symbol,
        direction=direction,
        score=75.0,
        tier=ScoreTier.MODERATE,
        position_size_percent=5.0,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward_ratio=2.0,
        components=ScoreComponents(
            sentiment_score=70.0,
            technical_score=80.0,
            sentiment_weight=0.4,
            technical_weight=0.6,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=1.0,
        ),
        reasoning="Test recommendation",
        timestamp=datetime.now(),
    )


def make_risk_result(
    approved: bool = True,
    quantity: int = 10,
) -> RiskCheckResult:
    """Create a test RiskCheckResult with sensible defaults."""
    return RiskCheckResult(
        approved=approved,
        adjusted_quantity=quantity if approved else 0,
        adjusted_value=quantity * 150.00 if approved else 0.0,
        rejection_reason=None if approved else "Risk check failed",
        warnings=[],
    )


@pytest.fixture
def mock_alpaca_client():
    """Mock AlpacaClient for testing."""
    client = Mock(spec=AlpacaClient)
    client.submit_order = AsyncMock()
    client.get_all_positions = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_risk_manager():
    """Mock RiskManager for testing."""
    manager = Mock(spec=RiskManager)
    manager.record_trade = Mock()
    manager.update_unrealized_pnl = Mock()
    return manager


@pytest.fixture
def trade_executor(mock_alpaca_client, mock_risk_manager):
    """Create TradeExecutor instance with mocked dependencies."""
    return TradeExecutor(
        alpaca_client=mock_alpaca_client,
        risk_manager=mock_risk_manager,
    )


@pytest.mark.asyncio
async def test_execute_approved_trade_returns_success(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that execute returns success for an approved trade."""
    # Arrange
    recommendation = make_recommendation(symbol="AAPL", direction=Direction.LONG)
    risk_result = make_risk_result(approved=True, quantity=10)

    # Mock Alpaca order response
    mock_alpaca_client.submit_order.return_value = {
        "id": "order_123",
        "status": "filled",
        "symbol": "AAPL",
        "qty": 10,
        "side": "buy",
        "type": "market",
        "filled_qty": 10,
        "filled_avg_price": 150.00,
    }

    # Act
    result = await trade_executor.execute(recommendation, risk_result)

    # Assert
    assert result.success is True
    assert result.order_id == "order_123"
    assert result.symbol == "AAPL"
    assert result.side == "buy"
    assert result.quantity == 10
    assert result.filled_price == 150.00
    assert result.error_message is None

    # Verify Alpaca client was called correctly
    mock_alpaca_client.submit_order.assert_called_once_with(
        symbol="AAPL",
        qty=10,
        side="buy",
        order_type="market",
    )

    # Verify risk manager methods were called
    mock_risk_manager.record_trade.assert_called_once_with("AAPL", 10, 150.00)
    mock_risk_manager.update_unrealized_pnl.assert_called_once_with(0.0)


@pytest.mark.asyncio
async def test_execute_rejected_trade_returns_failure(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that execute returns failure for a rejected trade."""
    # Arrange
    recommendation = make_recommendation(symbol="AAPL")
    risk_result = make_risk_result(approved=False, quantity=0)

    # Act
    result = await trade_executor.execute(recommendation, risk_result)

    # Assert
    assert result.success is False
    assert result.order_id is None
    assert result.symbol == "AAPL"
    assert result.quantity == 0
    assert result.filled_price is None
    assert result.error_message == "Risk check failed"

    # Verify Alpaca client was NOT called
    mock_alpaca_client.submit_order.assert_not_called()

    # Verify risk manager methods were NOT called
    mock_risk_manager.record_trade.assert_not_called()
    mock_risk_manager.update_unrealized_pnl.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tracks_position_after_success(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that execute tracks position after successful execution."""
    # Arrange
    recommendation = make_recommendation(
        symbol="NVDA",
        direction=Direction.LONG,
        entry_price=500.00,
        stop_loss=490.00,
        take_profit=520.00,
    )
    risk_result = make_risk_result(approved=True, quantity=5)

    # Mock Alpaca order response
    mock_alpaca_client.submit_order.return_value = {
        "id": "order_456",
        "status": "filled",
        "symbol": "NVDA",
        "qty": 5,
        "side": "buy",
        "type": "market",
        "filled_qty": 5,
        "filled_avg_price": 501.00,
    }

    # Act
    result = await trade_executor.execute(recommendation, risk_result)

    # Assert - verify execution result
    assert result.success is True
    assert result.symbol == "NVDA"

    # Assert - verify position is tracked
    tracked_positions = trade_executor.get_tracked_positions()
    assert "NVDA" in tracked_positions

    position = tracked_positions["NVDA"]
    assert position.symbol == "NVDA"
    assert position.quantity == 5
    assert position.entry_price == 501.00
    assert position.stop_loss == 490.00
    assert position.take_profit == 520.00
    assert position.order_id == "order_456"
    assert position.direction == Direction.LONG


@pytest.mark.asyncio
async def test_execute_short_direction(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that execute correctly handles SHORT direction trades."""
    # Arrange
    recommendation = make_recommendation(
        symbol="TSLA",
        direction=Direction.SHORT,
        entry_price=200.00,
        stop_loss=205.00,
        take_profit=190.00,
    )
    risk_result = make_risk_result(approved=True, quantity=8)

    # Mock Alpaca order response
    mock_alpaca_client.submit_order.return_value = {
        "id": "order_789",
        "status": "filled",
        "symbol": "TSLA",
        "qty": 8,
        "side": "sell",
        "type": "market",
        "filled_qty": 8,
        "filled_avg_price": 199.50,
    }

    # Act
    result = await trade_executor.execute(recommendation, risk_result)

    # Assert
    assert result.success is True
    assert result.side == "sell"
    assert result.symbol == "TSLA"
    assert result.quantity == 8

    # Verify Alpaca client was called with "sell"
    mock_alpaca_client.submit_order.assert_called_once_with(
        symbol="TSLA",
        qty=8,
        side="sell",
        order_type="market",
    )

    # Verify position tracking
    tracked_positions = trade_executor.get_tracked_positions()
    assert "TSLA" in tracked_positions
    position = tracked_positions["TSLA"]
    assert position.direction == Direction.SHORT
    assert position.entry_price == 199.50
