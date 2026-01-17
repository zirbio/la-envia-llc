# src/notifications/telegram_notifier.py
"""Telegram notification sender."""

import logging

from telegram import Bot

from execution.models import ExecutionResult
from risk.models import DailyRiskState
from scoring.models import TradeRecommendation

from .alert_formatter import AlertFormatter
from .models import Alert, AlertType
from .settings import NotificationSettings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends alerts via Telegram.

    Attributes:
        _settings: Notification settings.
        _formatter: Alert formatter.
        _bot: Telegram bot instance.
    """

    def __init__(
        self,
        settings: NotificationSettings,
        formatter: AlertFormatter,
    ):
        """Initialize the notifier.

        Args:
            settings: Notification settings.
            formatter: Alert formatter for message formatting.
        """
        self._settings = settings
        self._formatter = formatter
        self._bot: Bot | None = None

    async def start(self) -> None:
        """Initialize the Telegram bot."""
        if not self.is_enabled:
            logger.info("Telegram notifications disabled")
            return

        self._bot = Bot(token=self._settings.telegram_token)
        logger.info("Telegram notifier started")

    async def stop(self) -> None:
        """Shutdown the bot gracefully."""
        self._bot = None
        logger.info("Telegram notifier stopped")

    async def send_alert(self, alert: Alert) -> bool:
        """Send a generic alert.

        Args:
            alert: The alert to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.is_enabled:
            return False

        if self._bot is None:
            logger.warning("Bot not initialized, cannot send alert")
            return False

        # Check if this alert type is enabled
        if alert.alert_type.value not in self._settings.alert_types:
            logger.debug(f"Alert type {alert.alert_type.value} not enabled")
            return False

        try:
            await self._bot.send_message(
                chat_id=self._settings.chat_id,
                text=alert.message,
                parse_mode="HTML",
            )
            logger.info(f"Sent {alert.alert_type.value} alert")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    @property
    def is_enabled(self) -> bool:
        """Check if notifications are enabled and configured."""
        return self._settings.enabled and self._settings.is_configured

    async def send_new_signal(self, recommendation: TradeRecommendation) -> bool:
        """Send a new signal alert.

        Args:
            recommendation: The trade recommendation.

        Returns:
            True if sent successfully.
        """
        message = self._formatter.format_new_signal(recommendation)
        alert = Alert(
            alert_type=AlertType.NEW_SIGNAL,
            symbol=recommendation.symbol,
            message=message,
        )
        return await self.send_alert(alert)

    async def send_execution(
        self, result: ExecutionResult, is_entry: bool
    ) -> bool:
        """Send an execution alert.

        Args:
            result: The execution result.
            is_entry: True for entry, False for exit.

        Returns:
            True if sent successfully.
        """
        message = self._formatter.format_execution(result, is_entry)
        alert_type = AlertType.ENTRY_EXECUTED if is_entry else AlertType.EXIT_EXECUTED
        alert = Alert(
            alert_type=alert_type,
            symbol=result.symbol,
            message=message,
        )
        return await self.send_alert(alert)

    async def send_circuit_breaker(
        self, reason: str, risk_state: DailyRiskState
    ) -> bool:
        """Send a circuit breaker alert.

        Args:
            reason: Why the circuit breaker was triggered.
            risk_state: Current risk state.

        Returns:
            True if sent successfully.
        """
        message = self._formatter.format_circuit_breaker(reason, risk_state)
        alert = Alert(
            alert_type=AlertType.CIRCUIT_BREAKER,
            message=message,
        )
        return await self.send_alert(alert)

    async def send_daily_summary(self, stats: dict) -> bool:
        """Send daily trading summary.

        Args:
            stats: Dictionary with daily statistics.

        Returns:
            True if sent successfully.
        """
        message = self._formatter.format_daily_summary(stats)
        alert = Alert(
            alert_type=AlertType.DAILY_SUMMARY,
            message=message,
        )
        return await self.send_alert(alert)
