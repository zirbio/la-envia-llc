"""Tests for technical validator settings."""

import pytest
from pydantic import ValidationError

from src.config.settings import (
    Settings,
    TechnicalValidatorSettings,
    ValidatorsSettings,
)


class TestTechnicalValidatorSettings:
    """Tests for TechnicalValidatorSettings."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = TechnicalValidatorSettings()

        assert settings.enabled is True
        assert settings.rsi_period == 14
        assert settings.rsi_overbought == 70.0
        assert settings.rsi_oversold == 30.0
        assert settings.macd_fast == 12
        assert settings.macd_slow == 26
        assert settings.macd_signal == 9
        assert settings.stoch_k_period == 14
        assert settings.stoch_d_period == 3
        assert settings.adx_period == 14
        assert settings.adx_trend_threshold == 20.0
        assert settings.options_volume_spike_ratio == 2.0
        assert settings.iv_rank_warning_threshold == 80.0
        assert settings.veto_mode is True
        assert settings.lookback_bars == 50
        assert settings.timeframe == "5Min"

    def test_rsi_overbought_validation(self):
        """Test that rsi_overbought must be >= 50."""
        # Valid case: >= 50
        settings = TechnicalValidatorSettings(rsi_overbought=50.0)
        assert settings.rsi_overbought == 50.0

        settings = TechnicalValidatorSettings(rsi_overbought=75.0)
        assert settings.rsi_overbought == 75.0

        # Invalid case: < 50
        with pytest.raises(ValidationError) as exc_info:
            TechnicalValidatorSettings(rsi_overbought=45.0)

        assert "rsi_overbought" in str(exc_info.value)

    def test_rsi_oversold_validation(self):
        """Test that rsi_oversold must be <= 50."""
        # Valid case: <= 50
        settings = TechnicalValidatorSettings(rsi_oversold=50.0)
        assert settings.rsi_oversold == 50.0

        settings = TechnicalValidatorSettings(rsi_oversold=25.0)
        assert settings.rsi_oversold == 25.0

        # Invalid case: > 50
        with pytest.raises(ValidationError) as exc_info:
            TechnicalValidatorSettings(rsi_oversold=55.0)

        assert "rsi_oversold" in str(exc_info.value)

    def test_lookback_bars_validation(self):
        """Test that lookback_bars must be >= 20."""
        # Valid case: >= 20
        settings = TechnicalValidatorSettings(lookback_bars=20)
        assert settings.lookback_bars == 20

        settings = TechnicalValidatorSettings(lookback_bars=100)
        assert settings.lookback_bars == 100

        # Invalid case: < 20
        with pytest.raises(ValidationError) as exc_info:
            TechnicalValidatorSettings(lookback_bars=15)

        assert "lookback_bars" in str(exc_info.value)

    def test_rsi_period_range(self):
        """Test RSI period validation."""
        # Valid cases
        TechnicalValidatorSettings(rsi_period=2)
        TechnicalValidatorSettings(rsi_period=50)

        # Invalid cases
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(rsi_period=1)

        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(rsi_period=51)

    def test_macd_parameters_range(self):
        """Test MACD parameters validation."""
        # Valid cases
        TechnicalValidatorSettings(macd_fast=5, macd_slow=20, macd_signal=5)
        TechnicalValidatorSettings(macd_fast=20, macd_slow=50, macd_signal=15)

        # Invalid cases
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(macd_fast=4)

        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(macd_slow=51)

        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(macd_signal=4)

    def test_stochastic_parameters_range(self):
        """Test Stochastic parameters validation."""
        # Valid cases
        TechnicalValidatorSettings(stoch_k_period=5, stoch_d_period=2)
        TechnicalValidatorSettings(stoch_k_period=30, stoch_d_period=10)

        # Invalid cases
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(stoch_k_period=4)

        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(stoch_d_period=1)

    def test_adx_parameters_range(self):
        """Test ADX parameters validation."""
        # Valid cases
        TechnicalValidatorSettings(adx_period=5, adx_trend_threshold=10.0)
        TechnicalValidatorSettings(adx_period=30, adx_trend_threshold=40.0)

        # Invalid cases
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(adx_period=4)

        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(adx_trend_threshold=9.0)

    def test_volume_spike_ratio_range(self):
        """Test options volume spike ratio validation."""
        # Valid cases
        TechnicalValidatorSettings(options_volume_spike_ratio=1.0)
        TechnicalValidatorSettings(options_volume_spike_ratio=10.0)

        # Invalid cases
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(options_volume_spike_ratio=0.5)

        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(options_volume_spike_ratio=11.0)

    def test_iv_rank_threshold_range(self):
        """Test IV rank warning threshold validation."""
        # Valid cases
        TechnicalValidatorSettings(iv_rank_warning_threshold=50.0)
        TechnicalValidatorSettings(iv_rank_warning_threshold=100.0)

        # Invalid cases
        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(iv_rank_warning_threshold=49.0)

        with pytest.raises(ValidationError):
            TechnicalValidatorSettings(iv_rank_warning_threshold=101.0)


class TestValidatorsSettings:
    """Tests for ValidatorsSettings."""

    def test_default_technical_validator(self):
        """Test that technical validator is initialized with defaults."""
        validators = ValidatorsSettings()

        assert validators.technical is not None
        assert isinstance(validators.technical, TechnicalValidatorSettings)
        assert validators.technical.enabled is True

    def test_custom_technical_settings(self):
        """Test custom technical validator settings."""
        custom_technical = TechnicalValidatorSettings(
            enabled=False,
            rsi_period=20,
            lookback_bars=100,
        )

        validators = ValidatorsSettings(technical=custom_technical)

        assert validators.technical.enabled is False
        assert validators.technical.rsi_period == 20
        assert validators.technical.lookback_bars == 100


class TestSettingsIntegration:
    """Tests for Settings integration with validators."""

    def test_settings_has_validators_config(self):
        """Test that Settings includes validators configuration."""
        settings = Settings()

        assert hasattr(settings, "validators")
        assert isinstance(settings.validators, ValidatorsSettings)
        assert isinstance(settings.validators.technical, TechnicalValidatorSettings)

    def test_settings_validators_defaults(self):
        """Test that validators have correct defaults in Settings."""
        settings = Settings()

        assert settings.validators.technical.enabled is True
        assert settings.validators.technical.rsi_period == 14
        assert settings.validators.technical.veto_mode is True

    def test_settings_from_dict(self):
        """Test Settings creation from dictionary."""
        config_dict = {
            "validators": {
                "technical": {
                    "enabled": False,
                    "rsi_period": 20,
                    "lookback_bars": 100,
                }
            }
        }

        settings = Settings(**config_dict)

        assert settings.validators.technical.enabled is False
        assert settings.validators.technical.rsi_period == 20
        assert settings.validators.technical.lookback_bars == 100
        # Verify defaults are still applied for unspecified fields
        assert settings.validators.technical.macd_fast == 12

    def test_settings_from_yaml_includes_validators(self):
        """Test that Settings loaded from YAML includes validator config."""
        from pathlib import Path

        yaml_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        settings = Settings.from_yaml(yaml_path)

        # Verify validators config is loaded
        assert settings.validators is not None
        assert settings.validators.technical is not None
        assert settings.validators.technical.enabled is True
        assert settings.validators.technical.rsi_period == 14
        assert settings.validators.technical.rsi_overbought == 70.0
        assert settings.validators.technical.rsi_oversold == 30.0
        assert settings.validators.technical.veto_mode is True
        assert settings.validators.technical.lookback_bars == 50
        assert settings.validators.technical.timeframe == "5Min"
