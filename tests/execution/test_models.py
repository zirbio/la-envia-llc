# tests/execution/test_models.py
"""Tests for execution data models."""
from datetime import datetime, timezone

import pytest

from src.execution.models import ExecutionResult, TrackedPosition
from src.scoring.models import Direction


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_execution_result(self):
        """Test creating a successful execution result."""
        timestamp = datetime.now(timezone.utc)
        result = ExecutionResult(
            success=True,
            order_id="abc123",
            symbol="AAPL",
            side="buy",
            quantity=10,
            filled_price=150.50,
            error_message=None,
            timestamp=timestamp,
        )

        assert result.success is True
        assert result.order_id == "abc123"
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.quantity == 10
        assert result.filled_price == 150.50
        assert result.error_message is None
        assert result.timestamp == timestamp

    def test_failed_execution_result(self):
        """Test creating a failed execution result."""
        timestamp = datetime.now(timezone.utc)
        result = ExecutionResult(
            success=False,
            order_id=None,
            symbol="TSLA",
            side="sell",
            quantity=5,
            filled_price=None,
            error_message="Insufficient buying power",
            timestamp=timestamp,
        )

        assert result.success is False
        assert result.order_id is None
        assert result.symbol == "TSLA"
        assert result.side == "sell"
        assert result.quantity == 5
        assert result.filled_price is None
        assert result.error_message == "Insufficient buying power"
        assert result.timestamp == timestamp

    def test_execution_result_has_timestamp(self):
        """Test that execution result captures timestamp."""
        before = datetime.now(timezone.utc)
        result = ExecutionResult(
            success=True,
            order_id="xyz789",
            symbol="GOOGL",
            side="buy",
            quantity=3,
            filled_price=2800.00,
            error_message=None,
            timestamp=datetime.now(timezone.utc),
        )
        after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after
        assert isinstance(result.timestamp, datetime)


class TestTrackedPosition:
    """Tests for TrackedPosition dataclass."""

    def test_tracked_position_long(self):
        """Test creating a tracked long position."""
        entry_time = datetime.now(timezone.utc)
        position = TrackedPosition(
            symbol="AAPL",
            quantity=10,
            entry_price=150.00,
            entry_time=entry_time,
            stop_loss=145.00,
            take_profit=160.00,
            order_id="order123",
            direction=Direction.LONG,
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 10
        assert position.entry_price == 150.00
        assert position.entry_time == entry_time
        assert position.stop_loss == 145.00
        assert position.take_profit == 160.00
        assert position.order_id == "order123"
        assert position.direction == Direction.LONG

    def test_tracked_position_short(self):
        """Test creating a tracked short position."""
        entry_time = datetime.now(timezone.utc)
        position = TrackedPosition(
            symbol="TSLA",
            quantity=5,
            entry_price=800.00,
            entry_time=entry_time,
            stop_loss=820.00,
            take_profit=760.00,
            order_id="order456",
            direction=Direction.SHORT,
        )

        assert position.symbol == "TSLA"
        assert position.quantity == 5
        assert position.entry_price == 800.00
        assert position.entry_time == entry_time
        assert position.stop_loss == 820.00
        assert position.take_profit == 760.00
        assert position.order_id == "order456"
        assert position.direction == Direction.SHORT
