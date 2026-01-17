"""Dashboard state management."""
from datetime import datetime
from typing import ClassVar

from src.dashboard.models import AlertEvent, AlertLevel, AlertType


class DashboardState:
    """Singleton state manager for the dashboard.

    Maintains alert history and provides connection points
    for system managers.
    """

    _instance: ClassVar["DashboardState | None"] = None

    def __init__(self) -> None:
        """Initialize dashboard state."""
        self._alerts: list[AlertEvent] = []
        self._max_alerts: int = 50

    @classmethod
    def get_instance(cls) -> "DashboardState":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    @property
    def alerts(self) -> list[AlertEvent]:
        """Get all alerts, most recent first."""
        return self._alerts

    @property
    def unread_count(self) -> int:
        """Count of unread alerts."""
        return sum(1 for alert in self._alerts if not alert.read)

    def add_alert(
        self,
        alert_type: AlertType,
        level: AlertLevel,
        title: str,
        message: str,
        symbol: str | None = None,
    ) -> None:
        """Add a new alert to history."""
        alert = AlertEvent(
            timestamp=datetime.now(),
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            symbol=symbol,
        )
        self._alerts.insert(0, alert)

        # Trim to max alerts
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[: self._max_alerts]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts = []

    def mark_all_read(self) -> None:
        """Mark all alerts as read."""
        for alert in self._alerts:
            alert.read = True
