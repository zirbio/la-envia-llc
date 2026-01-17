# tests/validation/test_settings.py
import pytest
from src.validation.settings import ValidationSettings


class TestValidationSettings:
    def test_default_values(self):
        settings = ValidationSettings()
        assert settings.scenario_timeout_seconds == 30
        assert settings.mock_market_delay_ms == 100
        assert settings.fail_fast is True

    def test_custom_values(self):
        settings = ValidationSettings(
            scenario_timeout_seconds=60,
            mock_market_delay_ms=50,
            fail_fast=False,
        )
        assert settings.scenario_timeout_seconds == 60
        assert settings.mock_market_delay_ms == 50
        assert settings.fail_fast is False

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValueError):
            ValidationSettings(scenario_timeout_seconds=0)

    def test_delay_must_be_non_negative(self):
        with pytest.raises(ValueError):
            ValidationSettings(mock_market_delay_ms=-1)
