"""Data models for the dashboard."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AlertType(Enum):
    """Types of alerts."""

    TRADE_EXECUTED = "trade_executed"
    CIRCUIT_BREAKER = "circuit_breaker"
    GATE_CHANGE = "gate_change"
    SYSTEM_ERROR = "system_error"


class AlertLevel(Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class AlertEvent:
    """An alert event for the dashboard."""

    timestamp: datetime
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    symbol: str | None = None
    read: bool = field(default=False)
