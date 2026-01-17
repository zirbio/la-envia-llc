# tests/notifications/test_integration.py
"""Integration tests for notifications module."""

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNotificationIntegration:
    """Integration tests for full notification flow."""

    @pytest.mark.asyncio
    async def test_full_signal_notification_flow(self):
        """Test complete flow: signal -> format -> send."""
        from notifications import (
            AlertFormatter,
            NotificationSettings,
            TelegramNotifier,
        )
        from scoring.models import (
            Direction,
            ScoreComponents,
            ScoreTier,
            TradeRecommendation,
        )

        # Setup
        settings = NotificationSettings(
            enabled=True,
            telegram_token="test_token",
            chat_id="12345",
        )
        formatter = AlertFormatter()
        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        # Create recommendation
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
            reasoning="High sentiment confidence",
            timestamp=datetime.now(),
        )

        # Send notification
        result = await notifier.send_new_signal(rec)

        # Verify
        assert result is True
        mock_bot.send_message.assert_called_once()
        sent_text = mock_bot.send_message.call_args.kwargs["text"]
        assert "AAPL" in sent_text
        assert "LONG" in sent_text
        assert "85" in sent_text
        assert "178.50" in sent_text

    @pytest.mark.asyncio
    async def test_circuit_breaker_notification_flow(self):
        """Test circuit breaker notification flow."""
        from notifications import (
            AlertFormatter,
            NotificationSettings,
            TelegramNotifier,
        )
        from risk.models import DailyRiskState

        settings = NotificationSettings(
            enabled=True,
            telegram_token="test_token",
            chat_id="12345",
        )
        formatter = AlertFormatter()
        notifier = TelegramNotifier(settings=settings, formatter=formatter)

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

        result = await notifier.send_circuit_breaker("Daily loss limit", state)

        assert result is True
        sent_text = mock_bot.send_message.call_args.kwargs["text"]
        assert "CIRCUIT BREAKER" in sent_text
        assert "450" in sent_text

    @pytest.mark.asyncio
    async def test_disabled_alert_type_not_sent(self):
        """Alert types not in settings.alert_types are not sent."""
        from notifications import (
            AlertFormatter,
            AlertType,
            Alert,
            NotificationSettings,
            TelegramNotifier,
        )

        settings = NotificationSettings(
            enabled=True,
            telegram_token="test_token",
            chat_id="12345",
            alert_types=["new_signal"],  # Only new_signal enabled
        )
        formatter = AlertFormatter()
        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(return_value=MagicMock())
        notifier._bot = mock_bot

        # Try to send circuit breaker (not enabled)
        alert = Alert(
            alert_type=AlertType.CIRCUIT_BREAKER,
            message="Test circuit breaker",
        )
        result = await notifier.send_alert(alert)

        # Should return False and not send
        assert result is False
        mock_bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_checklist_flow(self):
        """Test checklist send -> check items -> complete flow."""
        from notifications import ChecklistHandler, NotificationSettings

        settings = NotificationSettings(
            checklist_items=["Item 1", "Item 2", "Item 3"]
        )

        mock_bot = MagicMock()
        mock_message = MagicMock()
        mock_message.message_id = 999
        mock_bot.send_message = AsyncMock(return_value=mock_message)
        mock_bot.edit_message_text = AsyncMock()

        handler = ChecklistHandler(
            settings=settings,
            bot=mock_bot,
            chat_id="12345",
        )

        # Send checklist
        await handler.send_checklist()
        assert handler._message_id == 999
        assert handler.is_checklist_complete() is False

        # Check items one by one
        await handler.on_item_checked(0)
        assert handler.is_checklist_complete() is False

        await handler.on_item_checked(1)
        assert handler.is_checklist_complete() is False

        await handler.on_item_checked(2)
        assert handler.is_checklist_complete() is True

        # Verify edit_message_text was called for each check
        assert mock_bot.edit_message_text.call_count == 3

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_telegram_error(self):
        """Notifier handles Telegram errors gracefully."""
        from notifications import (
            AlertFormatter,
            AlertType,
            Alert,
            NotificationSettings,
            TelegramNotifier,
        )

        settings = NotificationSettings(
            enabled=True,
            telegram_token="test_token",
            chat_id="12345",
        )
        formatter = AlertFormatter()
        notifier = TelegramNotifier(settings=settings, formatter=formatter)

        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock(
            side_effect=Exception("Telegram API error")
        )
        notifier._bot = mock_bot

        alert = Alert(
            alert_type=AlertType.NEW_SIGNAL,
            message="Test message",
        )

        # Should not raise, just return False
        result = await notifier.send_alert(alert)
        assert result is False
