# tests/config/test_risk_settings.py
"""Tests for RiskSettings configuration."""

import pytest
from pydantic import ValidationError

from src.config.settings import RiskSettings


class TestRiskSettings:
    """Test suite for RiskSettings."""

    def test_default_values(self):
        """Verify default values are set correctly."""
        settings = RiskSettings()

        assert settings.enabled is True
        assert settings.max_position_value == 1000.0
        assert settings.max_daily_loss == 500.0
        assert settings.unrealized_warning_threshold == 300.0

    def test_custom_values(self):
        """Verify custom values can be set."""
        settings = RiskSettings(
            enabled=False,
            max_position_value=2000.0,
            max_daily_loss=1000.0,
            unrealized_warning_threshold=600.0
        )

        assert settings.enabled is False
        assert settings.max_position_value == 2000.0
        assert settings.max_daily_loss == 1000.0
        assert settings.unrealized_warning_threshold == 600.0

    def test_max_position_value_must_be_positive(self):
        """Verify max_position_value must be greater than 0."""
        # Test with zero - should fail
        with pytest.raises(ValidationError) as exc_info:
            RiskSettings(max_position_value=0)

        assert "greater than 0" in str(exc_info.value).lower()

        # Test with negative - should fail
        with pytest.raises(ValidationError) as exc_info:
            RiskSettings(max_position_value=-100.0)

        assert "greater than 0" in str(exc_info.value).lower()

        # Test with positive - should succeed
        settings = RiskSettings(max_position_value=500.0)
        assert settings.max_position_value == 500.0

    def test_max_daily_loss_must_be_positive(self):
        """Verify max_daily_loss must be greater than 0."""
        # Test with zero - should fail
        with pytest.raises(ValidationError) as exc_info:
            RiskSettings(max_daily_loss=0)

        assert "greater than 0" in str(exc_info.value).lower()

        # Test with negative - should fail
        with pytest.raises(ValidationError) as exc_info:
            RiskSettings(max_daily_loss=-50.0)

        assert "greater than 0" in str(exc_info.value).lower()

        # Test with positive - should succeed
        settings = RiskSettings(max_daily_loss=250.0)
        assert settings.max_daily_loss == 250.0

    def test_unrealized_warning_threshold_must_be_positive(self):
        """Verify unrealized_warning_threshold must be greater than 0."""
        # Test with zero - should fail
        with pytest.raises(ValidationError) as exc_info:
            RiskSettings(unrealized_warning_threshold=0)

        assert "greater than 0" in str(exc_info.value).lower()

        # Test with negative - should fail
        with pytest.raises(ValidationError) as exc_info:
            RiskSettings(unrealized_warning_threshold=-100.0)

        assert "greater than 0" in str(exc_info.value).lower()

        # Test with positive - should succeed
        settings = RiskSettings(unrealized_warning_threshold=150.0)
        assert settings.unrealized_warning_threshold == 150.0
