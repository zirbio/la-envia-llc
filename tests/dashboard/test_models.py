"""Tests for dashboard models."""
from datetime import datetime

from src.dashboard.models import AlertEvent, AlertLevel, AlertType


class TestAlertEvent:
    """Tests for AlertEvent model."""

    def test_create_alert_event(self) -> None:
        """Should create an alert event with all fields."""
        event = AlertEvent(
            timestamp=datetime(2026, 1, 17, 9, 30, 0),
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Trade Executed",
            message="Bought 100 NVDA @ $140.00",
            symbol="NVDA",
        )

        assert event.alert_type == AlertType.TRADE_EXECUTED
        assert event.level == AlertLevel.INFO
        assert event.symbol == "NVDA"
        assert not event.read

    def test_alert_types(self) -> None:
        """Should have all expected alert types."""
        assert AlertType.TRADE_EXECUTED.value == "trade_executed"
        assert AlertType.CIRCUIT_BREAKER.value == "circuit_breaker"
        assert AlertType.GATE_CHANGE.value == "gate_change"
        assert AlertType.SYSTEM_ERROR.value == "system_error"

    def test_alert_levels(self) -> None:
        """Should have all expected alert levels."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
