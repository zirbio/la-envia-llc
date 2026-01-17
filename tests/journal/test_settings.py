# tests/journal/test_settings.py
"""Tests for journal settings."""
import pytest

from src.journal.settings import JournalSettings


class TestJournalSettings:
    """Tests for JournalSettings."""

    def test_default_settings(self):
        """Default settings should have sensible values."""
        settings = JournalSettings()

        assert settings.enabled is True
        assert settings.data_dir == "data/trades"
        assert settings.auto_log_entries is True
        assert settings.auto_log_exits is True
        assert settings.default_period_days == 30
        assert settings.weekly_report_enabled is True
        assert settings.weekly_report_day == "saturday"

    def test_custom_settings(self):
        """Settings should accept custom values."""
        settings = JournalSettings(
            enabled=False,
            data_dir="custom/path",
            default_period_days=60,
        )

        assert settings.enabled is False
        assert settings.data_dir == "custom/path"
        assert settings.default_period_days == 60

    def test_weekly_report_day_validation(self):
        """weekly_report_day should only accept valid days."""
        with pytest.raises(ValueError):
            JournalSettings(weekly_report_day="invalid")

    def test_valid_weekday_names(self):
        """All weekday names should be valid."""
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            settings = JournalSettings(weekly_report_day=day)
            assert settings.weekly_report_day == day
