# tests/config/test_scoring_settings.py
import pytest
from pydantic import ValidationError
from src.config.settings import Settings, ScoringSettings


class TestScoringSettings:
    """Tests for ScoringSettings configuration."""

    def test_settings_has_scoring_config(self):
        """Verify AppSettings has scoring field."""
        settings = Settings()
        assert hasattr(settings, "scoring")
        assert isinstance(settings.scoring, ScoringSettings)

    def test_default_values(self):
        """Verify all defaults are correct."""
        settings = ScoringSettings()

        # Enabled
        assert settings.enabled is True

        # Score thresholds
        assert settings.tier_strong_threshold == 80
        assert settings.tier_moderate_threshold == 60
        assert settings.tier_weak_threshold == 40

        # Position sizing
        assert settings.position_size_strong == 100.0
        assert settings.position_size_moderate == 50.0
        assert settings.position_size_weak == 25.0

        # Trade parameters
        assert settings.default_stop_loss_percent == 2.0
        assert settings.default_risk_reward_ratio == 2.0

        # Dynamic weights
        assert settings.base_sentiment_weight == 0.5
        assert settings.base_technical_weight == 0.5
        assert settings.strong_trend_adx == 30.0
        assert settings.weak_trend_adx == 20.0

        # Confluence
        assert settings.confluence_window_minutes == 15
        assert settings.confluence_bonus_2_signals == 0.10
        assert settings.confluence_bonus_3_signals == 0.20

        # Time factors
        assert settings.premarket_factor == 0.9
        assert settings.afterhours_factor == 0.8
        assert settings.earnings_proximity_days == 3
        assert settings.earnings_factor == 0.7

        # Source credibility
        assert settings.credibility_tier1_multiplier == 1.2
        assert settings.credibility_tier2_multiplier == 1.0
        assert settings.credibility_tier3_multiplier == 0.8
        assert settings.tier1_sources == ["unusual_whales", "optionsflow"]

    def test_threshold_validation(self):
        """Verify ge/le constraints work for thresholds."""
        # tier_strong_threshold: ge=0, le=100
        with pytest.raises(ValidationError):
            ScoringSettings(tier_strong_threshold=-1)
        with pytest.raises(ValidationError):
            ScoringSettings(tier_strong_threshold=101)

        # tier_moderate_threshold: ge=0, le=100
        with pytest.raises(ValidationError):
            ScoringSettings(tier_moderate_threshold=-1)
        with pytest.raises(ValidationError):
            ScoringSettings(tier_moderate_threshold=101)

        # tier_weak_threshold: ge=0, le=100
        with pytest.raises(ValidationError):
            ScoringSettings(tier_weak_threshold=-1)
        with pytest.raises(ValidationError):
            ScoringSettings(tier_weak_threshold=101)

        # Valid boundary values should work
        valid = ScoringSettings(
            tier_strong_threshold=100,
            tier_moderate_threshold=0,
            tier_weak_threshold=50,
        )
        assert valid.tier_strong_threshold == 100
        assert valid.tier_moderate_threshold == 0
        assert valid.tier_weak_threshold == 50

    def test_position_size_validation(self):
        """Verify position size constraints."""
        # position_size_strong: ge=0, le=100
        with pytest.raises(ValidationError):
            ScoringSettings(position_size_strong=-1)
        with pytest.raises(ValidationError):
            ScoringSettings(position_size_strong=101)

        # position_size_moderate: ge=0, le=100
        with pytest.raises(ValidationError):
            ScoringSettings(position_size_moderate=-0.1)
        with pytest.raises(ValidationError):
            ScoringSettings(position_size_moderate=100.1)

        # position_size_weak: ge=0, le=100
        with pytest.raises(ValidationError):
            ScoringSettings(position_size_weak=-10)
        with pytest.raises(ValidationError):
            ScoringSettings(position_size_weak=150)

        # Valid boundary values should work
        valid = ScoringSettings(
            position_size_strong=100,
            position_size_moderate=0,
            position_size_weak=50,
        )
        assert valid.position_size_strong == 100.0
        assert valid.position_size_moderate == 0.0
        assert valid.position_size_weak == 50.0

    def test_stop_loss_validation(self):
        """Verify stop loss constraints (ge=0.1, le=10.0)."""
        with pytest.raises(ValidationError):
            ScoringSettings(default_stop_loss_percent=0.05)
        with pytest.raises(ValidationError):
            ScoringSettings(default_stop_loss_percent=10.5)

        valid = ScoringSettings(default_stop_loss_percent=5.0)
        assert valid.default_stop_loss_percent == 5.0

    def test_risk_reward_validation(self):
        """Verify risk reward ratio constraints (ge=1.0, le=5.0)."""
        with pytest.raises(ValidationError):
            ScoringSettings(default_risk_reward_ratio=0.5)
        with pytest.raises(ValidationError):
            ScoringSettings(default_risk_reward_ratio=6.0)

        valid = ScoringSettings(default_risk_reward_ratio=3.0)
        assert valid.default_risk_reward_ratio == 3.0

    def test_weight_validation(self):
        """Verify weight constraints (ge=0, le=1)."""
        # base_sentiment_weight
        with pytest.raises(ValidationError):
            ScoringSettings(base_sentiment_weight=-0.1)
        with pytest.raises(ValidationError):
            ScoringSettings(base_sentiment_weight=1.1)

        # base_technical_weight
        with pytest.raises(ValidationError):
            ScoringSettings(base_technical_weight=-0.1)
        with pytest.raises(ValidationError):
            ScoringSettings(base_technical_weight=1.1)

        valid = ScoringSettings(base_sentiment_weight=0.7, base_technical_weight=0.3)
        assert valid.base_sentiment_weight == 0.7
        assert valid.base_technical_weight == 0.3

    def test_adx_validation(self):
        """Verify ADX constraints."""
        # strong_trend_adx: ge=20, le=50
        with pytest.raises(ValidationError):
            ScoringSettings(strong_trend_adx=19)
        with pytest.raises(ValidationError):
            ScoringSettings(strong_trend_adx=51)

        # weak_trend_adx: ge=10, le=30
        with pytest.raises(ValidationError):
            ScoringSettings(weak_trend_adx=9)
        with pytest.raises(ValidationError):
            ScoringSettings(weak_trend_adx=31)

    def test_confluence_validation(self):
        """Verify confluence constraints."""
        # confluence_window_minutes: ge=1, le=60
        with pytest.raises(ValidationError):
            ScoringSettings(confluence_window_minutes=0)
        with pytest.raises(ValidationError):
            ScoringSettings(confluence_window_minutes=61)

        # confluence_bonus: ge=0, le=0.5
        with pytest.raises(ValidationError):
            ScoringSettings(confluence_bonus_2_signals=-0.1)
        with pytest.raises(ValidationError):
            ScoringSettings(confluence_bonus_2_signals=0.6)
        with pytest.raises(ValidationError):
            ScoringSettings(confluence_bonus_3_signals=-0.1)
        with pytest.raises(ValidationError):
            ScoringSettings(confluence_bonus_3_signals=0.6)

    def test_time_factor_validation(self):
        """Verify time factor constraints."""
        # premarket_factor: ge=0.5, le=1.0
        with pytest.raises(ValidationError):
            ScoringSettings(premarket_factor=0.4)
        with pytest.raises(ValidationError):
            ScoringSettings(premarket_factor=1.1)

        # afterhours_factor: ge=0.5, le=1.0
        with pytest.raises(ValidationError):
            ScoringSettings(afterhours_factor=0.4)
        with pytest.raises(ValidationError):
            ScoringSettings(afterhours_factor=1.1)

        # earnings_proximity_days: ge=1, le=14
        with pytest.raises(ValidationError):
            ScoringSettings(earnings_proximity_days=0)
        with pytest.raises(ValidationError):
            ScoringSettings(earnings_proximity_days=15)

        # earnings_factor: ge=0.5, le=1.0
        with pytest.raises(ValidationError):
            ScoringSettings(earnings_factor=0.4)
        with pytest.raises(ValidationError):
            ScoringSettings(earnings_factor=1.1)

    def test_credibility_multiplier_validation(self):
        """Verify credibility multiplier constraints."""
        # tier1: ge=1.0, le=1.5
        with pytest.raises(ValidationError):
            ScoringSettings(credibility_tier1_multiplier=0.9)
        with pytest.raises(ValidationError):
            ScoringSettings(credibility_tier1_multiplier=1.6)

        # tier2: ge=0.8, le=1.2
        with pytest.raises(ValidationError):
            ScoringSettings(credibility_tier2_multiplier=0.7)
        with pytest.raises(ValidationError):
            ScoringSettings(credibility_tier2_multiplier=1.3)

        # tier3: ge=0.5, le=1.0
        with pytest.raises(ValidationError):
            ScoringSettings(credibility_tier3_multiplier=0.4)
        with pytest.raises(ValidationError):
            ScoringSettings(credibility_tier3_multiplier=1.1)

    def test_custom_tier1_sources(self):
        """Verify tier1_sources can be customized."""
        custom_sources = ["source1", "source2", "source3"]
        settings = ScoringSettings(tier1_sources=custom_sources)
        assert settings.tier1_sources == custom_sources
