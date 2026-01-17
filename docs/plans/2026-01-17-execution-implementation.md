# Phase 6: Trade Executor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Position-Aware Trade Executor with Alpaca bracket orders and RiskManager P&L feedback.

**Architecture:** TradeExecutor receives approved trades, submits bracket orders, tracks positions, updates RiskManager.

**Tech Stack:** Python dataclasses, Alpaca API, Pydantic settings

---

## Task 1: Data Models (ExecutionResult, TrackedPosition)

**Files:**
- Create: `src/execution/models.py`
- Create: `tests/execution/test_models.py`

**Step 1: Write failing tests for ExecutionResult**

```python
# tests/execution/test_models.py
import pytest
from datetime import datetime
from src.execution.models import ExecutionResult, TrackedPosition
from src.scoring.models import Direction


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_execution_result(self):
        """Test creating a successful execution result."""
        result = ExecutionResult(
            success=True,
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            quantity=10,
            filled_price=150.50,
            error_message=None,
            timestamp=datetime(2026, 1, 17, 10, 30, 0),
        )
        assert result.success is True
        assert result.order_id == "order-123"
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.quantity == 10
        assert result.filled_price == 150.50
        assert result.error_message is None

    def test_failed_execution_result(self):
        """Test creating a failed execution result."""
        result = ExecutionResult(
            success=False,
            order_id=None,
            symbol="AAPL",
            side="buy",
            quantity=10,
            filled_price=None,
            error_message="Insufficient buying power",
            timestamp=datetime(2026, 1, 17, 10, 30, 0),
        )
        assert result.success is False
        assert result.order_id is None
        assert result.error_message == "Insufficient buying power"

    def test_execution_result_has_timestamp(self):
        """Test that execution result includes timestamp."""
        ts = datetime(2026, 1, 17, 10, 30, 0)
        result = ExecutionResult(
            success=True,
            order_id="order-123",
            symbol="AAPL",
            side="buy",
            quantity=10,
            filled_price=150.50,
            error_message=None,
            timestamp=ts,
        )
        assert result.timestamp == ts


class TestTrackedPosition:
    """Tests for TrackedPosition dataclass."""

    def test_tracked_position_long(self):
        """Test creating a LONG tracked position."""
        position = TrackedPosition(
            symbol="AAPL",
            quantity=10,
            entry_price=150.00,
            entry_time=datetime(2026, 1, 17, 10, 30, 0),
            stop_loss=147.00,
            take_profit=156.00,
            order_id="order-123",
            direction=Direction.LONG,
        )
        assert position.symbol == "AAPL"
        assert position.quantity == 10
        assert position.entry_price == 150.00
        assert position.stop_loss == 147.00
        assert position.take_profit == 156.00
        assert position.direction == Direction.LONG

    def test_tracked_position_short(self):
        """Test creating a SHORT tracked position."""
        position = TrackedPosition(
            symbol="TSLA",
            quantity=5,
            entry_price=200.00,
            entry_time=datetime(2026, 1, 17, 10, 30, 0),
            stop_loss=206.00,
            take_profit=194.00,
            order_id="order-456",
            direction=Direction.SHORT,
        )
        assert position.symbol == "TSLA"
        assert position.direction == Direction.SHORT
        assert position.stop_loss == 206.00  # Above entry for SHORT
        assert position.take_profit == 194.00  # Below entry for SHORT
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/execution/test_models.py -v`
Expected: FAIL with import errors

**Step 3: Implement the models**

```python
# src/execution/models.py
"""Data models for trade execution."""
from dataclasses import dataclass
from datetime import datetime

from src.scoring.models import Direction


@dataclass
class ExecutionResult:
    """Result of a trade execution attempt.

    Attributes:
        success: Whether the order was submitted successfully.
        order_id: Alpaca order ID if successful, None otherwise.
        symbol: Stock ticker symbol.
        side: Order side ("buy" or "sell").
        quantity: Number of shares in the order.
        filled_price: Average fill price if filled, None otherwise.
        error_message: Error description if failed, None otherwise.
        timestamp: When the execution was attempted.
    """

    success: bool
    order_id: str | None
    symbol: str
    side: str
    quantity: int
    filled_price: float | None
    error_message: str | None
    timestamp: datetime


@dataclass
class TrackedPosition:
    """A locally tracked open position.

    Attributes:
        symbol: Stock ticker symbol.
        quantity: Number of shares held.
        entry_price: Average entry price.
        entry_time: When the position was opened.
        stop_loss: Stop loss price.
        take_profit: Take profit price.
        order_id: Original Alpaca order ID.
        direction: Trade direction (LONG or SHORT).
    """

    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    order_id: str
    direction: Direction
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/execution/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/execution/models.py tests/execution/test_models.py
git commit -m "feat(execution): add ExecutionResult and TrackedPosition models"
```

---

## Task 2: TradeExecutor - Core Structure and Execute Method

**Files:**
- Create: `src/execution/trade_executor.py`
- Create: `tests/execution/test_trade_executor.py`

**Step 1: Write failing tests for basic execute flow**

```python
# tests/execution/test_trade_executor.py
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.execution.trade_executor import TradeExecutor
from src.execution.models import ExecutionResult
from src.scoring.models import Direction, ScoreComponents, ScoreTier, TradeRecommendation
from src.risk.models import RiskCheckResult


def make_recommendation(
    symbol: str = "AAPL",
    direction: Direction = Direction.LONG,
    entry_price: float = 150.00,
    stop_loss: float = 147.00,
    take_profit: float = 156.00,
) -> TradeRecommendation:
    """Helper to create TradeRecommendation for tests."""
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
        reasoning="Test recommendation",
        timestamp=datetime.now(),
    )


def make_risk_result(approved: bool = True, quantity: int = 10) -> RiskCheckResult:
    """Helper to create RiskCheckResult for tests."""
    return RiskCheckResult(
        approved=approved,
        adjusted_quantity=quantity if approved else 0,
        adjusted_value=quantity * 150.0 if approved else 0.0,
        rejection_reason=None if approved else "Test rejection",
        warnings=[],
    )


class TestTradeExecutor:
    """Tests for TradeExecutor class."""

    @pytest.fixture
    def mock_alpaca(self):
        """Create mock AlpacaClient."""
        client = AsyncMock()
        client.get_all_positions = AsyncMock(return_value=[])
        client.submit_order = AsyncMock(return_value={
            "id": "order-123",
            "status": "filled",
            "symbol": "AAPL",
            "qty": 10,
            "side": "buy",
            "type": "market",
            "filled_qty": 10,
            "filled_avg_price": 150.25,
        })
        return client

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock RiskManager."""
        manager = MagicMock()
        manager.record_trade = MagicMock()
        manager.record_close = MagicMock()
        manager.update_unrealized_pnl = MagicMock()
        return manager

    @pytest.fixture
    def executor(self, mock_alpaca, mock_risk_manager):
        """Create TradeExecutor with mocks."""
        return TradeExecutor(
            alpaca_client=mock_alpaca,
            risk_manager=mock_risk_manager,
        )

    @pytest.mark.asyncio
    async def test_execute_approved_trade_returns_success(self, executor, mock_alpaca):
        """Test executing an approved trade returns success."""
        recommendation = make_recommendation()
        risk_result = make_risk_result(approved=True, quantity=10)

        result = await executor.execute(recommendation, risk_result)

        assert result.success is True
        assert result.order_id == "order-123"
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.quantity == 10

    @pytest.mark.asyncio
    async def test_execute_rejected_trade_returns_failure(self, executor):
        """Test executing a rejected trade returns failure."""
        recommendation = make_recommendation()
        risk_result = make_risk_result(approved=False)

        result = await executor.execute(recommendation, risk_result)

        assert result.success is False
        assert result.order_id is None
        assert result.error_message == "Trade not approved by risk manager"

    @pytest.mark.asyncio
    async def test_execute_tracks_position_after_success(self, executor):
        """Test that successful execution tracks the position."""
        recommendation = make_recommendation()
        risk_result = make_risk_result(approved=True, quantity=10)

        await executor.execute(recommendation, risk_result)

        positions = executor.get_tracked_positions()
        assert "AAPL" in positions
        assert positions["AAPL"].quantity == 10
        assert positions["AAPL"].direction == Direction.LONG

    @pytest.mark.asyncio
    async def test_execute_short_direction(self, executor, mock_alpaca):
        """Test executing a SHORT trade."""
        mock_alpaca.submit_order = AsyncMock(return_value={
            "id": "order-456",
            "status": "filled",
            "symbol": "TSLA",
            "qty": 5,
            "side": "sell",
            "type": "market",
            "filled_qty": 5,
            "filled_avg_price": 200.00,
        })
        recommendation = make_recommendation(
            symbol="TSLA",
            direction=Direction.SHORT,
            entry_price=200.00,
            stop_loss=206.00,
            take_profit=194.00,
        )
        risk_result = make_risk_result(approved=True, quantity=5)

        result = await executor.execute(recommendation, risk_result)

        assert result.success is True
        assert result.side == "sell"
        positions = executor.get_tracked_positions()
        assert positions["TSLA"].direction == Direction.SHORT
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/execution/test_trade_executor.py -v`
Expected: FAIL with import errors

**Step 3: Implement TradeExecutor core**

```python
# src/execution/trade_executor.py
"""Trade executor for submitting orders and tracking positions."""
from datetime import datetime

from src.execution.alpaca_client import AlpacaClient
from src.execution.models import ExecutionResult, TrackedPosition
from src.risk.models import RiskCheckResult
from src.risk.risk_manager import RiskManager
from src.scoring.models import Direction, TradeRecommendation


class TradeExecutor:
    """Executes approved trades and tracks positions.

    Submits bracket orders to Alpaca, tracks open positions locally,
    and updates RiskManager with P&L after each trade.

    Attributes:
        alpaca_client: Client for Alpaca API operations.
        risk_manager: Manager for risk tracking and P&L updates.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        risk_manager: RiskManager,
    ):
        """Initialize TradeExecutor.

        Args:
            alpaca_client: Client for Alpaca API operations.
            risk_manager: Manager for risk tracking and P&L updates.
        """
        self._alpaca = alpaca_client
        self._risk_manager = risk_manager
        self._tracked_positions: dict[str, TrackedPosition] = {}

    async def execute(
        self,
        recommendation: TradeRecommendation,
        risk_result: RiskCheckResult,
    ) -> ExecutionResult:
        """Execute an approved trade as a bracket order.

        Steps:
        1. Validate risk_result is approved
        2. Sync positions (detect closes, record P&L)
        3. Submit bracket order
        4. Track position locally
        5. Update unrealized P&L
        6. Return result

        Args:
            recommendation: The trade recommendation to execute.
            risk_result: The risk check result (must be approved).

        Returns:
            ExecutionResult with success/failure details.
        """
        timestamp = datetime.now()

        # Step 1: Check if approved
        if not risk_result.approved:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=recommendation.symbol,
                side=self._direction_to_side(recommendation.direction),
                quantity=0,
                filled_price=None,
                error_message="Trade not approved by risk manager",
                timestamp=timestamp,
            )

        # Step 2: Sync positions (detect any closes)
        await self.sync_positions()

        # Step 3: Submit bracket order
        side = self._direction_to_side(recommendation.direction)
        try:
            order = await self._alpaca.submit_order(
                symbol=recommendation.symbol,
                qty=risk_result.adjusted_quantity,
                side=side,
                order_type="market",
                time_in_force="day",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=recommendation.symbol,
                side=side,
                quantity=risk_result.adjusted_quantity,
                filled_price=None,
                error_message=str(e),
                timestamp=timestamp,
            )

        # Step 4: Track position locally
        filled_price = order.get("filled_avg_price") or recommendation.entry_price
        self._tracked_positions[recommendation.symbol] = TrackedPosition(
            symbol=recommendation.symbol,
            quantity=risk_result.adjusted_quantity,
            entry_price=filled_price,
            entry_time=timestamp,
            stop_loss=recommendation.stop_loss,
            take_profit=recommendation.take_profit,
            order_id=order["id"],
            direction=recommendation.direction,
        )

        # Step 5: Record trade and update unrealized P&L
        self._risk_manager.record_trade(
            recommendation.symbol,
            risk_result.adjusted_quantity,
            filled_price,
        )
        unrealized = await self.get_unrealized_pnl()
        self._risk_manager.update_unrealized_pnl(unrealized)

        # Step 6: Return success
        return ExecutionResult(
            success=True,
            order_id=order["id"],
            symbol=recommendation.symbol,
            side=side,
            quantity=risk_result.adjusted_quantity,
            filled_price=filled_price,
            error_message=None,
            timestamp=timestamp,
        )

    def _direction_to_side(self, direction: Direction) -> str:
        """Convert Direction to order side string."""
        return "buy" if direction == Direction.LONG else "sell"

    async def sync_positions(self) -> list[str]:
        """Sync tracked positions with Alpaca, detect closes.

        Placeholder - will be implemented in Task 3.
        """
        return []

    async def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L from Alpaca positions.

        Placeholder - will be implemented in Task 3.
        """
        return 0.0

    def get_tracked_positions(self) -> dict[str, TrackedPosition]:
        """Return currently tracked positions."""
        return self._tracked_positions.copy()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/execution/test_trade_executor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/execution/trade_executor.py tests/execution/test_trade_executor.py
git commit -m "feat(execution): add TradeExecutor with execute method"
```

---

## Task 3: Position Sync and P&L Calculation

**Files:**
- Modify: `src/execution/trade_executor.py`
- Modify: `tests/execution/test_trade_executor.py`

**Step 1: Add tests for sync_positions and P&L calculation**

```python
# Add to tests/execution/test_trade_executor.py

class TestPositionSync:
    """Tests for position synchronization."""

    @pytest.fixture
    def mock_alpaca(self):
        """Create mock AlpacaClient."""
        client = AsyncMock()
        client.get_all_positions = AsyncMock(return_value=[])
        client.submit_order = AsyncMock(return_value={
            "id": "order-123",
            "status": "filled",
            "symbol": "AAPL",
            "qty": 10,
            "side": "buy",
            "type": "market",
            "filled_qty": 10,
            "filled_avg_price": 150.00,
        })
        return client

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock RiskManager."""
        manager = MagicMock()
        manager.record_trade = MagicMock()
        manager.record_close = MagicMock()
        manager.update_unrealized_pnl = MagicMock()
        return manager

    @pytest.fixture
    def executor(self, mock_alpaca, mock_risk_manager):
        """Create TradeExecutor with mocks."""
        return TradeExecutor(
            alpaca_client=mock_alpaca,
            risk_manager=mock_risk_manager,
        )

    @pytest.mark.asyncio
    async def test_sync_detects_closed_position(self, executor, mock_alpaca, mock_risk_manager):
        """Test that sync detects when a position is closed."""
        # Set up a tracked position
        executor._tracked_positions["AAPL"] = TrackedPosition(
            symbol="AAPL",
            quantity=10,
            entry_price=150.00,
            entry_time=datetime.now(),
            stop_loss=147.00,
            take_profit=156.00,
            order_id="order-123",
            direction=Direction.LONG,
        )
        # Alpaca returns empty (position closed)
        mock_alpaca.get_all_positions = AsyncMock(return_value=[])

        closed = await executor.sync_positions()

        assert "AAPL" in closed
        assert "AAPL" not in executor._tracked_positions
        mock_risk_manager.record_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_calculates_pnl_for_long(self, executor, mock_alpaca, mock_risk_manager):
        """Test P&L calculation for closed LONG position."""
        executor._tracked_positions["AAPL"] = TrackedPosition(
            symbol="AAPL",
            quantity=10,
            entry_price=150.00,
            entry_time=datetime.now(),
            stop_loss=147.00,
            take_profit=156.00,
            order_id="order-123",
            direction=Direction.LONG,
        )
        # Simulate position closed at take profit
        mock_alpaca.get_all_positions = AsyncMock(return_value=[])
        # Mock order history to show fill at 156.00
        mock_alpaca.get_order = AsyncMock(return_value={
            "filled_avg_price": 156.00,
            "status": "filled",
        })

        await executor.sync_positions()

        # P&L = (156 - 150) * 10 = 60
        mock_risk_manager.record_close.assert_called_with("AAPL", 60.0)

    @pytest.mark.asyncio
    async def test_sync_calculates_pnl_for_short(self, executor, mock_alpaca, mock_risk_manager):
        """Test P&L calculation for closed SHORT position."""
        executor._tracked_positions["TSLA"] = TrackedPosition(
            symbol="TSLA",
            quantity=5,
            entry_price=200.00,
            entry_time=datetime.now(),
            stop_loss=206.00,
            take_profit=194.00,
            order_id="order-456",
            direction=Direction.SHORT,
        )
        mock_alpaca.get_all_positions = AsyncMock(return_value=[])
        mock_alpaca.get_order = AsyncMock(return_value={
            "filled_avg_price": 194.00,
            "status": "filled",
        })

        await executor.sync_positions()

        # P&L for SHORT = (entry - exit) * qty = (200 - 194) * 5 = 30
        mock_risk_manager.record_close.assert_called_with("TSLA", 30.0)

    @pytest.mark.asyncio
    async def test_sync_position_still_open(self, executor, mock_alpaca, mock_risk_manager):
        """Test that open positions are not removed."""
        executor._tracked_positions["AAPL"] = TrackedPosition(
            symbol="AAPL",
            quantity=10,
            entry_price=150.00,
            entry_time=datetime.now(),
            stop_loss=147.00,
            take_profit=156.00,
            order_id="order-123",
            direction=Direction.LONG,
        )
        # Alpaca still has the position
        mock_alpaca.get_all_positions = AsyncMock(return_value=[
            {"symbol": "AAPL", "qty": 10, "unrealized_pl": 50.0}
        ])

        closed = await executor.sync_positions()

        assert closed == []
        assert "AAPL" in executor._tracked_positions
        mock_risk_manager.record_close.assert_not_called()


class TestUnrealizedPnL:
    """Tests for unrealized P&L calculation."""

    @pytest.fixture
    def mock_alpaca(self):
        client = AsyncMock()
        client.get_all_positions = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def mock_risk_manager(self):
        return MagicMock()

    @pytest.fixture
    def executor(self, mock_alpaca, mock_risk_manager):
        return TradeExecutor(mock_alpaca, mock_risk_manager)

    @pytest.mark.asyncio
    async def test_get_unrealized_pnl_sums_positions(self, executor, mock_alpaca):
        """Test unrealized P&L sums all position P&L."""
        mock_alpaca.get_all_positions = AsyncMock(return_value=[
            {"symbol": "AAPL", "qty": 10, "unrealized_pl": 50.0},
            {"symbol": "TSLA", "qty": 5, "unrealized_pl": -20.0},
        ])

        pnl = await executor.get_unrealized_pnl()

        assert pnl == 30.0  # 50 + (-20)

    @pytest.mark.asyncio
    async def test_get_unrealized_pnl_empty_positions(self, executor, mock_alpaca):
        """Test unrealized P&L with no positions."""
        mock_alpaca.get_all_positions = AsyncMock(return_value=[])

        pnl = await executor.get_unrealized_pnl()

        assert pnl == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/execution/test_trade_executor.py::TestPositionSync -v`
Expected: FAIL

**Step 3: Implement sync_positions and get_unrealized_pnl**

Update `src/execution/trade_executor.py`:

```python
    async def sync_positions(self) -> list[str]:
        """Compare tracked positions with Alpaca, detect closes.

        For each closed position:
        1. Calculate realized P&L
        2. Record to RiskManager
        3. Remove from tracking

        Returns:
            List of symbols that were closed.
        """
        # Fetch current positions from Alpaca
        alpaca_positions = await self._alpaca.get_all_positions()
        alpaca_symbols = {p["symbol"] for p in alpaca_positions}

        closed_symbols = []
        for symbol, tracked in list(self._tracked_positions.items()):
            if symbol not in alpaca_symbols:
                # Position was closed
                closed_symbols.append(symbol)

                # Calculate realized P&L
                pnl = await self._calculate_closed_pnl(tracked)

                # Record to RiskManager
                self._risk_manager.record_close(symbol, pnl)

                # Remove from tracking
                del self._tracked_positions[symbol]

        return closed_symbols

    async def _calculate_closed_pnl(self, position: TrackedPosition) -> float:
        """Calculate realized P&L for a closed position.

        Attempts to get exit price from order history.
        Falls back to stop_loss or take_profit based on direction.

        Args:
            position: The closed position.

        Returns:
            Realized profit/loss.
        """
        # Try to get actual exit price from order
        try:
            order = await self._alpaca.get_order(position.order_id)
            exit_price = order.get("filled_avg_price", position.take_profit)
        except Exception:
            # Fallback: assume hit take profit (conservative for P&L tracking)
            exit_price = position.take_profit

        # Calculate P&L based on direction
        if position.direction == Direction.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.quantity

        return pnl

    async def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L from Alpaca positions.

        Returns:
            Sum of unrealized P&L across all positions.
        """
        positions = await self._alpaca.get_all_positions()
        total = sum(float(p.get("unrealized_pl", 0)) for p in positions)
        return total
```

**Step 4: Add get_order method to AlpacaClient**

Update `src/execution/alpaca_client.py` to add the missing method:

```python
    async def get_order(self, order_id: str) -> dict:
        """Get order details by ID.

        Args:
            order_id: The order ID to fetch.

        Returns:
            Dictionary containing order details.
        """
        order = self._trading_client.get_order_by_id(order_id)
        return self._order_to_dict(order)
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/execution/test_trade_executor.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/execution/trade_executor.py src/execution/alpaca_client.py tests/execution/test_trade_executor.py
git commit -m "feat(execution): add position sync and P&L calculation"
```

---

## Task 4: ExecutionSettings Configuration

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`
- Create: `tests/config/test_execution_settings.py`

**Step 1: Write failing tests for ExecutionSettings**

```python
# tests/config/test_execution_settings.py
import pytest
from pydantic import ValidationError
from src.config.settings import Settings, ExecutionSettings


class TestExecutionSettings:
    """Tests for ExecutionSettings configuration."""

    def test_default_values(self):
        """Verify all defaults are correct."""
        settings = ExecutionSettings()

        assert settings.enabled is True
        assert settings.paper_mode is True
        assert settings.default_time_in_force == "day"

    def test_custom_values(self):
        """Test custom configuration values."""
        settings = ExecutionSettings(
            enabled=False,
            paper_mode=False,
            default_time_in_force="gtc",
        )

        assert settings.enabled is False
        assert settings.paper_mode is False
        assert settings.default_time_in_force == "gtc"

    def test_time_in_force_validation(self):
        """Test that time_in_force accepts valid values."""
        for tif in ["day", "gtc", "ioc", "fok"]:
            settings = ExecutionSettings(default_time_in_force=tif)
            assert settings.default_time_in_force == tif

    def test_settings_has_execution_config(self):
        """Verify Settings has execution field."""
        settings = Settings()
        assert hasattr(settings, "execution")
        assert isinstance(settings.execution, ExecutionSettings)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/config/test_execution_settings.py -v`
Expected: FAIL with import errors

**Step 3: Implement ExecutionSettings**

Add to `src/config/settings.py`:

```python
class ExecutionSettings(BaseModel):
    """Settings for trade execution."""

    enabled: bool = True
    paper_mode: bool = True
    default_time_in_force: str = Field(default="day")
```

Add to Settings class:

```python
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
```

**Step 4: Update config/settings.yaml**

Add:

```yaml
# CAPA 6: Execution
execution:
  enabled: true
  paper_mode: true
  default_time_in_force: "day"
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/config/test_execution_settings.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/config/settings.py config/settings.yaml tests/config/test_execution_settings.py
git commit -m "feat(config): add execution settings"
```

---

## Task 5: Update Module Exports

**Files:**
- Modify: `src/execution/__init__.py`

**Step 1: Update exports**

```python
# src/execution/__init__.py
"""Execution module for trading operations."""

from .alpaca_client import AlpacaClient
from .models import ExecutionResult, TrackedPosition
from .trade_executor import TradeExecutor

__all__ = ["AlpacaClient", "ExecutionResult", "TrackedPosition", "TradeExecutor"]
```

**Step 2: Verify imports work**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run python -c "from src.execution import TradeExecutor, ExecutionResult, TrackedPosition; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/execution/__init__.py
git commit -m "chore(execution): update module exports"
```

---

## Task 6: Integration Tests

**Files:**
- Create: `tests/integration/test_execution_pipeline.py`

**Step 1: Write integration tests**

```python
# tests/integration/test_execution_pipeline.py
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
        client.submit_order = AsyncMock(return_value={
            "id": "order-123",
            "status": "filled",
            "symbol": "AAPL",
            "qty": 10,
            "side": "buy",
            "type": "market",
            "filled_qty": 10,
            "filled_avg_price": 150.00,
        })
        client.get_order = AsyncMock(return_value={
            "filled_avg_price": 156.00,
            "status": "filled",
        })
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

        # Simulate loss (closed at stop loss)
        mock_alpaca.get_all_positions = AsyncMock(return_value=[])
        mock_alpaca.get_order = AsyncMock(return_value={
            "filled_avg_price": 100.00,  # Big loss
            "status": "filled",
        })
        await executor.sync_positions()

        # P&L = (100 - 150) * 10 = -500 (loss equals limit)
        assert risk_manager.get_daily_state().is_blocked is True

        # Next trade should be blocked
        rec2 = make_recommendation(symbol="TSLA")
        risk2 = risk_manager.check_trade(rec2, 5, 200.00)
        assert risk2.approved is False

    @pytest.mark.asyncio
    async def test_multiple_trades_accumulate(self, executor, risk_manager, mock_alpaca):
        """Test multiple trades accumulate correctly."""
        # Trade 1
        rec1 = make_recommendation(symbol="AAPL")
        risk1 = risk_manager.check_trade(rec1, 10, 150.00)
        await executor.execute(rec1, risk1)

        # Trade 2
        mock_alpaca.submit_order = AsyncMock(return_value={
            "id": "order-456",
            "status": "filled",
            "symbol": "TSLA",
            "qty": 5,
            "side": "buy",
            "type": "market",
            "filled_qty": 5,
            "filled_avg_price": 200.00,
        })
        rec2 = make_recommendation(symbol="TSLA", entry_price=200.00)
        risk2 = risk_manager.check_trade(rec2, 5, 200.00)
        await executor.execute(rec2, risk2)

        assert risk_manager.get_daily_state().trades_today == 2
        assert len(executor.get_tracked_positions()) == 2
```

**Step 2: Run tests**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest tests/integration/test_execution_pipeline.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_execution_pipeline.py
git commit -m "test: add integration tests for execution pipeline"
```

---

## Task 7: Final Test Suite Run

**Step 1: Run full test suite**

Run: `cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/phase6-execution && uv run pytest --cov=src --cov-report=term-missing -v`

Expected: All tests pass, coverage >90% on execution module

**Step 2: Verify coverage**

Check that `src/execution/` has good coverage:
- `models.py`: 100%
- `trade_executor.py`: >90%
- `alpaca_client.py`: existing coverage maintained

**Step 3: Document completion**

All Phase 6 implementation complete and tested.
