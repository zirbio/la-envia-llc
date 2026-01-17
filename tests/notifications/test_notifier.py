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


class TestTelegramNotifierHelpers:
    """Tests for helper methods."""

    @pytest.mark.asyncio
    async def test_send_new_signal(self):
        """send_new_signal formats and sends signal."""
        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier
        from scoring.models import (
            Direction,
            ScoreComponents,
            ScoreTier,
            TradeRecommendation,
        )

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        rec = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=5.0,
            entry_price=178.50,
            stop_loss=176.00,
            take_profit=183.00,
            risk_reward_ratio=1.8,
            components=ScoreComponents(
                sentiment_score=90.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.35,
                confluence_bonus=0.1,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="High sentiment",
            timestamp=datetime.now(),
        )

        result = await notifier.send_new_signal(rec)

        assert result is True
        call_args = mock_bot.send_message.call_args
        assert "AAPL" in call_args.kwargs["text"]
        assert "LONG" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_execution_entry(self):
        """send_execution sends entry execution."""
        from execution.models import ExecutionResult
        from notifications.alert_formatter import AlertFormatter
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

        result = ExecutionResult(
            success=True,
            order_id="order123",
            symbol="AAPL",
            side="buy",
            quantity=50,
            filled_price=178.52,
            error_message=None,
            timestamp=datetime.now(),
        )

        sent = await notifier.send_execution(result, is_entry=True)

        assert sent is True
        call_args = mock_bot.send_message.call_args
        assert "ENTRY" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_circuit_breaker(self):
        """send_circuit_breaker sends circuit breaker alert."""
        from datetime import date

        from notifications.alert_formatter import AlertFormatter
        from notifications.settings import NotificationSettings
        from notifications.telegram_notifier import TelegramNotifier
        from risk.models import DailyRiskState

        settings = NotificationSettings(
            telegram_token="token",
            chat_id="12345",
        )
        notifier = TelegramNotifier(settings=settings, formatter=AlertFormatter())

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        state = DailyRiskState(
            date=date.today(),
            realized_pnl=-450.0,
            unrealized_pnl=0.0,
            trades_today=5,
            is_blocked=True,
        )

        sent = await notifier.send_circuit_breaker("Daily limit", state)

        assert sent is True
        call_args = mock_bot.send_message.call_args
        assert "CIRCUIT BREAKER" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_send_daily_summary(self):
        """send_daily_summary sends summary."""
        from notifications.alert_formatter import AlertFormatter
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

        stats = {
            "date": "2026-01-17",
            "total_trades": 5,
            "winners": 3,
            "losers": 2,
            "gross_pnl": 285.0,
        }

        sent = await notifier.send_daily_summary(stats)

        assert sent is True
        call_args = mock_bot.send_message.call_args
        assert "DAILY SUMMARY" in call_args.kwargs["text"]