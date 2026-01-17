# tests/notifications/test_notifier.py
"""Tests for TelegramNotifier."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTelegramNotifierInit:
    """Tests for TelegramNotifier initialization."""

    def test_init_with_settings(self):
        """Notifier initializes with settings."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="test_token",
            chat_id="12345",
        )
        formatter = AlertFormatter()

        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        assert notifier._settings == settings
        assert notifier._formatter == formatter
        assert notifier._bot is None

    def test_is_enabled_false_when_disabled(self):
        """is_enabled returns False when disabled in settings."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            enabled=False,
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        assert notifier.is_enabled is False

    def test_is_enabled_false_when_not_configured(self):
        """is_enabled returns False when token/chat_id missing."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            enabled=True,
            telegram_token="",
            chat_id="",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        assert notifier.is_enabled is False

    def test_is_enabled_true_when_configured(self):
        """is_enabled returns True when enabled and configured."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            enabled=True,
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        assert notifier.is_enabled is True


class TestTelegramNotifierStartStop:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_bot(self):
        """start() creates bot instance."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="test_token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        with patch("notifications.telegram_notifier.Bot") as mock_bot_class:
            mock_bot = MagicMock()
            mock_bot_class.return_value = mock_bot

            await notifier.start()

            mock_bot_class.assert_called_once_with(token="test_token")
            assert notifier._bot is not None

    @pytest.mark.asyncio
    async def test_stop_clears_bot(self):
        """stop() clears bot instance."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="test_token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())
        notifier._bot = MagicMock()

        await notifier.stop()

        assert notifier._bot is None


class TestTelegramNotifierSendAlert:
    """Tests for send_alert method."""

    @pytest.mark.asyncio
    async def test_send_alert_returns_false_when_disabled(self):
        """send_alert returns False when notifier is disabled."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.models import Alert, AlertType
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(enabled=False)
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        alert = Alert(alert_type=AlertType.NEW_SIGNAL, message="test")
        result = await notifier.send_alert(alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_returns_true_on_success(self):
        """send_alert returns True when message sent successfully."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.models import Alert, AlertType
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        alert = Alert(alert_type=AlertType.NEW_SIGNAL, message="test message")
        result = await notifier.send_alert(alert)

        assert result is True
        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alert_returns_false_on_error(self):
        """send_alert returns False on error (graceful degradation)."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.models import Alert, AlertType
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(side_effect=Exception("Network error"))
        notifier._bot = mock_bot

        alert = Alert(alert_type=AlertType.NEW_SIGNAL, message="test")
        result = await notifier.send_alert(alert)

        assert result is False
