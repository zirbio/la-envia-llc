"""Data models for notifications."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AlertType(Enum):
    """Type of alert to send."""

    NEW_SIGNAL = "new_signal"
    ENTRY_EXECUTED = "entry_executed"
    EXIT_EXECUTED = "exit_executed"
    CIRCUIT_BREAKER = "circuit_breaker"
    DAILY_SUMMARY = "daily_summary"


@dataclass
class Alert:
    """An alert to be sent via Telegram.

    Attributes:
        alert_type: Type of alert.
        symbol: Stock symbol (if applicable).
        message: Pre-formatted message (if not using formatter).
        data: Additional data for formatting.
        timestamp: When the alert was created.
    """

    alert_type: AlertType
    symbol: str | None = None
    message: str = ""
    data: dict | None = None
    timestamp: datetime = field(default_factory=datetime.now)
