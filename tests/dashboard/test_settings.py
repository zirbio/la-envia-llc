"""Tests for DashboardSettings."""
import pytest
from src.dashboard.settings import DashboardSettings


class TestDashboardSettings:
    """Tests for DashboardSettings."""

    def test_default_settings(self) -> None:
        """Default settings should have sensible values."""
        settings = DashboardSettings()

        assert settings.refresh_interval_seconds == 5
        assert settings.max_signals_displayed == 10
        assert settings.max_alerts_displayed == 50
        assert settings.theme == "dark"

    def test_custom_settings(self) -> None:
        """Custom settings should override defaults."""
        settings = DashboardSettings(
            refresh_interval_seconds=10,
            max_signals_displayed=20,
            theme="light",
        )

        assert settings.refresh_interval_seconds == 10
        assert settings.max_signals_displayed == 20
        assert settings.theme == "light"

    def test_refresh_interval_validation(self) -> None:
        """Refresh interval must be positive."""
        with pytest.raises(ValueError):
            DashboardSettings(refresh_interval_seconds=0)

    def test_theme_validation(self) -> None:
        """Theme must be light or dark."""
        with pytest.raises(ValueError):
            DashboardSettings(theme="invalid")

    def test_max_signals_validation(self) -> None:
        """Max signals must be positive."""
        with pytest.raises(ValueError):
            DashboardSettings(max_signals_displayed=0)

    def test_max_alerts_validation(self) -> None:
        """Max alerts must be positive."""
        with pytest.raises(ValueError):
            DashboardSettings(max_alerts_displayed=0)
