# tests/notifications/test_settings.py
"""Tests for notification settings."""

import pytest


class TestNotificationSettings:
    """Tests for NotificationSettings."""

    def test_default_values(self):
        """Settings have correct defaults."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings()

        assert settings.enabled is True
        assert settings.telegram_token == ""
        assert settings.chat_id == ""
        assert "new_signal" in settings.alert_types
        assert "entry_executed" in settings.alert_types
        assert "circuit_breaker" in settings.alert_types
        assert settings.pre_market_checklist_enabled is True
        assert settings.pre_market_checklist_time == "09:00"
        assert len(settings.checklist_items) == 5

    def test_custom_values(self):
        """Settings accept custom values."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(
            enabled=False,
            telegram_token="test_token",
            chat_id="12345",
            alert_types=["new_signal"],
            pre_market_checklist_enabled=False,
        )

        assert settings.enabled is False
        assert settings.telegram_token == "test_token"
        assert settings.chat_id == "12345"
        assert settings.alert_types == ["new_signal"]
        assert settings.pre_market_checklist_enabled is False

    def test_is_configured_false_when_no_token(self):
        """is_configured returns False when token is empty."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(telegram_token="", chat_id="12345")

        assert settings.is_configured is False

    def test_is_configured_false_when_no_chat_id(self):
        """is_configured returns False when chat_id is empty."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(telegram_token="token", chat_id="")

        assert settings.is_configured is False

    def test_is_configured_true_when_both_set(self):
        """is_configured returns True when both token and chat_id are set."""
        from notifications.settings import NotificationSettings

        settings = NotificationSettings(telegram_token="token", chat_id="12345")

        assert settings.is_configured is True
