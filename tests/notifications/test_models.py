"""Tests for notification models."""

from datetime import datetime

import pytest


class TestAlertType:
    """Tests for AlertType enum."""

    def test_new_signal_value(self):
        """AlertType.NEW_SIGNAL has correct value."""
        from notifications.models import AlertType

        assert AlertType.NEW_SIGNAL.value == "new_signal"

    def test_entry_executed_value(self):
        """AlertType.ENTRY_EXECUTED has correct value."""
        from notifications.models import AlertType

        assert AlertType.ENTRY_EXECUTED.value == "entry_executed"

    def test_exit_executed_value(self):
        """AlertType.EXIT_EXECUTED has correct value."""
        from notifications.models import AlertType

        assert AlertType.EXIT_EXECUTED.value == "exit_executed"

    def test_circuit_breaker_value(self):
        """AlertType.CIRCUIT_BREAKER has correct value."""
        from notifications.models import AlertType

        assert AlertType.CIRCUIT_BREAKER.value == "circuit_breaker"

    def test_daily_summary_value(self):
        """AlertType.DAILY_SUMMARY has correct value."""
        from notifications.models import AlertType

        assert AlertType.DAILY_SUMMARY.value == "daily_summary"


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation_minimal(self):
        """Alert can be created with just alert_type."""
        from notifications.models import Alert, AlertType

        alert = Alert(alert_type=AlertType.NEW_SIGNAL)

        assert alert.alert_type == AlertType.NEW_SIGNAL
        assert alert.symbol is None
        assert alert.message == ""
        assert alert.data is None
        assert isinstance(alert.timestamp, datetime)

    def test_alert_creation_full(self):
        """Alert can be created with all fields."""
        from notifications.models import Alert, AlertType

        ts = datetime(2026, 1, 17, 10, 30, 0)
        alert = Alert(
            alert_type=AlertType.ENTRY_EXECUTED,
            symbol="AAPL",
            message="Entry executed",
            data={"price": 178.50},
            timestamp=ts,
        )

        assert alert.alert_type == AlertType.ENTRY_EXECUTED
        assert alert.symbol == "AAPL"
        assert alert.message == "Entry executed"
        assert alert.data == {"price": 178.50}
        assert alert.timestamp == ts
