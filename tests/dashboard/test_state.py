"""Tests for DashboardState."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.dashboard.models import AlertEvent, AlertLevel, AlertType
from src.dashboard.state import DashboardState


class TestDashboardState:
    """Tests for DashboardState."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        DashboardState.reset_instance()

    def test_singleton_pattern(self) -> None:
        """DashboardState should return same instance."""
        state1 = DashboardState.get_instance()
        state2 = DashboardState.get_instance()

        assert state1 is state2

    def test_add_alert(self) -> None:
        """Should add alerts to history."""
        state = DashboardState.get_instance()

        state.add_alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Test Alert",
            message="Test message",
        )

        assert len(state.alerts) == 1
        assert state.alerts[0].title == "Test Alert"

    def test_alerts_limited_by_max(self) -> None:
        """Should limit alerts to max_alerts setting."""
        state = DashboardState.get_instance()
        state._max_alerts = 3

        for i in range(5):
            state.add_alert(
                alert_type=AlertType.SYSTEM_ERROR,
                level=AlertLevel.ERROR,
                title=f"Alert {i}",
                message=f"Message {i}",
            )

        assert len(state.alerts) == 3
        # Most recent alerts kept
        assert state.alerts[0].title == "Alert 4"

    def test_clear_alerts(self) -> None:
        """Should clear all alerts."""
        state = DashboardState.get_instance()
        state.add_alert(
            alert_type=AlertType.SYSTEM_ERROR,
            level=AlertLevel.ERROR,
            title="Test",
            message="Test",
        )
        state.clear_alerts()

        assert len(state.alerts) == 0

    def test_unread_count(self) -> None:
        """Should count unread alerts."""
        state = DashboardState.get_instance()

        state.add_alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Alert 1",
            message="Message 1",
        )
        state.add_alert(
            alert_type=AlertType.TRADE_EXECUTED,
            level=AlertLevel.INFO,
            title="Alert 2",
            message="Message 2",
        )

        assert state.unread_count == 2

        state.alerts[0].read = True
        assert state.unread_count == 1
