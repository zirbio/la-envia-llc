# src/notifications/settings.py
"""Settings for notifications module."""

from pydantic import BaseModel, computed_field


class NotificationSettings(BaseModel):
    """Configuration for Telegram notifications.

    Attributes:
        enabled: Whether notifications are enabled.
        telegram_token: Bot token from BotFather.
        chat_id: Telegram chat ID to send messages to.
        alert_types: Which alert types to send.
        pre_market_checklist_enabled: Whether to send pre-market checklist.
        pre_market_checklist_time: Time to send checklist (HH:MM).
        checklist_items: Items for the pre-market checklist.
    """

    enabled: bool = True
    telegram_token: str = ""
    chat_id: str = ""

    alert_types: list[str] = [
        "new_signal",
        "entry_executed",
        "exit_executed",
        "circuit_breaker",
        "daily_summary",
    ]

    pre_market_checklist_enabled: bool = True
    pre_market_checklist_time: str = "09:00"
    checklist_items: list[str] = [
        "Economic calendar reviewed",
        "Overnight news checked",
        "Watchlist prepared",
        "Mental state: focused",
        "Risk parameters confirmed",
    ]

    @computed_field
    @property
    def is_configured(self) -> bool:
        """Check if Telegram credentials are configured."""
        return bool(self.telegram_token and self.chat_id)
