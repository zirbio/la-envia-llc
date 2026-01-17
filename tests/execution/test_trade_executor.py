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
from gate.models import GateCheckResult, GateStatus


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
    client.get_order = AsyncMock()
    return client


@pytest.fixture
def mock_risk_manager():
    """Mock RiskManager for testing."""
    manager = Mock(spec=RiskManager)
    manager.record_trade = Mock()
    manager.update_unrealized_pnl = Mock()
    manager.record_close = Mock()
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


# Task 3: Position Sync and P&L Calculation Tests


@pytest.mark.asyncio
async def test_sync_detects_closed_position(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that sync_positions detects when a tracked position has been closed."""
    # Arrange - add a tracked position manually
    tracked_position = TrackedPosition(
        symbol="AAPL",
        quantity=10,
        entry_price=150.00,
        entry_time=datetime.now(),
        stop_loss=147.00,
        take_profit=156.00,
        order_id="order_123",
        direction=Direction.LONG,
    )
    trade_executor._tracked_positions["AAPL"] = tracked_position

    # Mock Alpaca - position is no longer there (closed)
    mock_alpaca_client.get_all_positions.return_value = []

    # Mock order details for PnL calculation
    mock_alpaca_client.get_order = AsyncMock(return_value={
        "id": "order_123",
        "filled_avg_price": 156.00,  # Exited at take profit
    })

    # Act
    closed_symbols = await trade_executor.sync_positions()

    # Assert
    assert "AAPL" in closed_symbols
    assert "AAPL" not in trade_executor._tracked_positions

    # Verify record_close was called with calculated PnL
    # LONG: (156.00 - 150.00) * 10 = 60.00
    mock_risk_manager.record_close.assert_called_once_with("AAPL", 60.00)


@pytest.mark.asyncio
async def test_sync_calculates_pnl_for_long(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that sync_positions correctly calculates P&L for a closed LONG position."""
    # Arrange
    tracked_position = TrackedPosition(
        symbol="NVDA",
        quantity=5,
        entry_price=500.00,
        entry_time=datetime.now(),
        stop_loss=490.00,
        take_profit=520.00,
        order_id="order_456",
        direction=Direction.LONG,
    )
    trade_executor._tracked_positions["NVDA"] = tracked_position

    # Mock Alpaca - position is closed
    mock_alpaca_client.get_all_positions.return_value = []

    # Mock order exit at $515.00
    mock_alpaca_client.get_order = AsyncMock(return_value={
        "id": "order_456",
        "filled_avg_price": 515.00,
    })

    # Act
    closed_symbols = await trade_executor.sync_positions()

    # Assert
    assert "NVDA" in closed_symbols

    # LONG: (515.00 - 500.00) * 5 = 75.00
    mock_risk_manager.record_close.assert_called_once_with("NVDA", 75.00)


@pytest.mark.asyncio
async def test_sync_calculates_pnl_for_short(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that sync_positions correctly calculates P&L for a closed SHORT position."""
    # Arrange
    tracked_position = TrackedPosition(
        symbol="TSLA",
        quantity=8,
        entry_price=200.00,
        entry_time=datetime.now(),
        stop_loss=205.00,
        take_profit=190.00,
        order_id="order_789",
        direction=Direction.SHORT,
    )
    trade_executor._tracked_positions["TSLA"] = tracked_position

    # Mock Alpaca - position is closed
    mock_alpaca_client.get_all_positions.return_value = []

    # Mock order exit at $192.00 (didn't hit take profit)
    mock_alpaca_client.get_order = AsyncMock(return_value={
        "id": "order_789",
        "filled_avg_price": 192.00,
    })

    # Act
    closed_symbols = await trade_executor.sync_positions()

    # Assert
    assert "TSLA" in closed_symbols

    # SHORT: (200.00 - 192.00) * 8 = 64.00
    mock_risk_manager.record_close.assert_called_once_with("TSLA", 64.00)


@pytest.mark.asyncio
async def test_sync_position_still_open(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that sync_positions does not remove positions that are still open."""
    # Arrange
    tracked_position = TrackedPosition(
        symbol="AAPL",
        quantity=10,
        entry_price=150.00,
        entry_time=datetime.now(),
        stop_loss=147.00,
        take_profit=156.00,
        order_id="order_123",
        direction=Direction.LONG,
    )
    trade_executor._tracked_positions["AAPL"] = tracked_position

    # Mock Alpaca - position is still open
    mock_alpaca_client.get_all_positions.return_value = [
        {
            "symbol": "AAPL",
            "qty": 10,
            "unrealized_pl": 25.00,
        }
    ]

    # Act
    closed_symbols = await trade_executor.sync_positions()

    # Assert
    assert closed_symbols == []
    assert "AAPL" in trade_executor._tracked_positions
    mock_risk_manager.record_close.assert_not_called()


@pytest.mark.asyncio
async def test_get_unrealized_pnl_sums_positions(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that get_unrealized_pnl sums P&L from all Alpaca positions."""
    # Arrange
    mock_alpaca_client.get_all_positions.return_value = [
        {"symbol": "AAPL", "qty": 10, "unrealized_pl": 50.00},
        {"symbol": "NVDA", "qty": 5, "unrealized_pl": -20.00},
        {"symbol": "TSLA", "qty": 8, "unrealized_pl": 30.00},
    ]

    # Act
    total_pnl = await trade_executor.get_unrealized_pnl()

    # Assert
    assert total_pnl == 60.00  # 50.00 - 20.00 + 30.00


@pytest.mark.asyncio
async def test_get_unrealized_pnl_empty_positions(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that get_unrealized_pnl returns 0.0 when there are no positions."""
    # Arrange
    mock_alpaca_client.get_all_positions.return_value = []

    # Act
    total_pnl = await trade_executor.get_unrealized_pnl()

    # Assert
    assert total_pnl == 0.0


@pytest.mark.asyncio
async def test_sync_fallback_to_take_profit_on_order_error(
    trade_executor, mock_alpaca_client, mock_risk_manager
):
    """Test that sync_positions falls back to take_profit when order lookup fails."""
    # Arrange
    tracked_position = TrackedPosition(
        symbol="MSFT",
        quantity=12,
        entry_price=300.00,
        entry_time=datetime.now(),
        stop_loss=295.00,
        take_profit=310.00,
        order_id="order_999",
        direction=Direction.LONG,
    )
    trade_executor._tracked_positions["MSFT"] = tracked_position

    # Mock Alpaca - position is closed
    mock_alpaca_client.get_all_positions.return_value = []

    # Mock order lookup to raise an exception (e.g., order not found)
    mock_alpaca_client.get_order = AsyncMock(side_effect=Exception("Order not found"))

    # Act
    closed_symbols = await trade_executor.sync_positions()

    # Assert
    assert "MSFT" in closed_symbols

    # Should use take_profit as fallback: (310.00 - 300.00) * 12 = 120.00
    mock_risk_manager.record_close.assert_called_once_with("MSFT", 120.00)


# Task 9: Gate Integration Tests


class TestTradeExecutorGateIntegration:
    """Tests for gate integration in TradeExecutor."""

    @pytest.fixture
    def mock_gate_status_open(self) -> GateStatus:
        """Gate status with is_open=True."""
        return GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="test", passed=True, reason=None, data={})],
            position_size_factor=1.0,
        )

    @pytest.fixture
    def mock_gate_status_closed(self) -> GateStatus:
        """Gate status with is_open=False."""
        return GateStatus(
            timestamp=datetime.now(),
            is_open=False,
            checks=[GateCheckResult(name="vix", passed=False, reason="VIX too high", data={})],
            position_size_factor=0.0,
        )

    @pytest.fixture
    def mock_gate_status_elevated(self) -> GateStatus:
        """Gate status with elevated VIX (factor 0.5)."""
        return GateStatus(
            timestamp=datetime.now(),
            is_open=True,
            checks=[GateCheckResult(name="vix", passed=True, reason=None, data={"status": "elevated"})],
            position_size_factor=0.5,
        )

    @pytest.mark.asyncio
    async def test_execute_respects_gate_closed(
        self,
        trade_executor: TradeExecutor,
        mock_gate_status_closed: GateStatus,
    ) -> None:
        """Execute returns failure when gate is closed."""
        recommendation = make_recommendation(symbol="AAPL", direction=Direction.LONG)
        risk_result = make_risk_result(approved=True, quantity=100)

        result = await trade_executor.execute(
            recommendation, risk_result, gate_status=mock_gate_status_closed
        )

        assert result.success is False
        assert "gate closed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_applies_size_factor(
        self,
        mock_alpaca_client: Mock,
        mock_risk_manager: Mock,
        mock_gate_status_elevated: GateStatus,
    ) -> None:
        """Execute applies position_size_factor to quantity."""
        # Set up approved for 100 shares
        recommendation = make_recommendation(symbol="AAPL", direction=Direction.LONG)
        risk_result = make_risk_result(approved=True, quantity=100)

        mock_alpaca_client.submit_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 150.0,
            "filled_qty": 50,
        }

        executor = TradeExecutor(mock_alpaca_client, mock_risk_manager)
        result = await executor.execute(
            recommendation, risk_result, gate_status=mock_gate_status_elevated
        )

        # Verify order was submitted with reduced quantity
        call_args = mock_alpaca_client.submit_order.call_args
        assert call_args.kwargs["qty"] == 50  # 100 * 0.5

    @pytest.mark.asyncio
    async def test_execute_without_gate_status(
        self,
        trade_executor: TradeExecutor,
    ) -> None:
        """Execute works normally when gate_status not provided."""
        recommendation = make_recommendation(symbol="AAPL", direction=Direction.LONG)
        risk_result = make_risk_result(approved=True, quantity=10)

        # Mock the Alpaca client response
        trade_executor._alpaca.submit_order.return_value = {
            "id": "order-123",
            "filled_avg_price": 150.0,
            "filled_qty": 10,
        }

        result = await trade_executor.execute(recommendation, risk_result)

        assert result.success is True
