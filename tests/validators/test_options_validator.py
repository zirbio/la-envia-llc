# tests/validators/test_options_validator.py
"""Tests for options flow validator."""

import pytest

from src.validators.models import OptionsFlowData
from src.validators.options_validator import OptionsValidator


class TestOptionsValidator:
    """Test suite for OptionsValidator."""

    @pytest.fixture
    def validator(self) -> OptionsValidator:
        """Create a validator with default settings."""
        return OptionsValidator(
            volume_spike_ratio=2.0,
            iv_rank_warning_threshold=80.0,
        )

    def test_detect_volume_spike(self, validator: OptionsValidator) -> None:
        """Test detection of volume spike."""
        # Volume ratio >= 2.0 should trigger volume spike
        data = OptionsFlowData(
            volume_ratio=2.5,
            iv_rank=50.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )

        is_enhanced, warnings = validator.validate(data)

        assert is_enhanced is True
        assert len(warnings) == 0

    def test_high_iv_warning(self, validator: OptionsValidator) -> None:
        """Test high IV generates warning."""
        # IV rank > 80 should generate warning
        data = OptionsFlowData(
            volume_ratio=1.0,
            iv_rank=85.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )

        is_enhanced, warnings = validator.validate(data)

        assert is_enhanced is False
        assert len(warnings) == 1
        assert "high IV" in warnings[0].lower() or "implied volatility" in warnings[0].lower()

    def test_normal_options_flow(self, validator: OptionsValidator) -> None:
        """Test normal options flow without triggers."""
        # Normal conditions - no spike, no high IV
        data = OptionsFlowData(
            volume_ratio=1.2,
            iv_rank=50.0,
            put_call_ratio=0.9,
            unusual_activity=False,
        )

        is_enhanced, warnings = validator.validate(data)

        assert is_enhanced is False
        assert len(warnings) == 0

    def test_validate_returns_modifier(self, validator: OptionsValidator) -> None:
        """Test that validator calculates confidence modifier correctly."""
        # Test volume spike modifier
        data_spike = OptionsFlowData(
            volume_ratio=2.5,
            iv_rank=50.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )
        modifier_spike = validator.get_confidence_modifier(data_spike)
        assert modifier_spike > 1.0  # Should be boosted

        # Test high IV penalty
        data_high_iv = OptionsFlowData(
            volume_ratio=1.0,
            iv_rank=85.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )
        modifier_high_iv = validator.get_confidence_modifier(data_high_iv)
        assert modifier_high_iv < 1.0  # Should be penalized

    def test_unusual_activity_with_volume_spike(self, validator: OptionsValidator) -> None:
        """Test unusual activity combined with volume spike."""
        # Unusual activity alone should trigger enhancement
        data = OptionsFlowData(
            volume_ratio=1.5,
            iv_rank=50.0,
            put_call_ratio=0.8,
            unusual_activity=True,
        )

        is_enhanced, warnings = validator.validate(data)

        # Unusual activity should trigger enhancement
        assert is_enhanced is True
        assert len(warnings) == 0

    def test_confidence_modifier_cheap_options(self, validator: OptionsValidator) -> None:
        """Test confidence boost for cheap options (low IV)."""
        # Low IV rank (< 50) should boost confidence
        data = OptionsFlowData(
            volume_ratio=1.0,
            iv_rank=30.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )

        modifier = validator.get_confidence_modifier(data)
        assert modifier > 1.0

    def test_confidence_modifier_combined_boosts(self, validator: OptionsValidator) -> None:
        """Test multiple confidence boosts stack correctly."""
        # Volume spike + unusual activity + low IV should give maximum boost
        data = OptionsFlowData(
            volume_ratio=3.0,
            iv_rank=30.0,
            put_call_ratio=0.8,
            unusual_activity=True,
        )

        modifier = validator.get_confidence_modifier(data)

        # Base 1.0 + 0.1 (spike) + 0.1 (unusual) + 0.1 (low IV) = 1.3
        assert modifier == pytest.approx(1.3, abs=0.01)

    def test_confidence_modifier_clamped_to_range(self, validator: OptionsValidator) -> None:
        """Test confidence modifier is clamped to [0.8, 1.3]."""
        # Maximum boost scenario
        data_max = OptionsFlowData(
            volume_ratio=5.0,
            iv_rank=20.0,
            put_call_ratio=0.8,
            unusual_activity=True,
        )
        modifier_max = validator.get_confidence_modifier(data_max)
        assert modifier_max <= 1.3

        # Minimum penalty scenario
        data_min = OptionsFlowData(
            volume_ratio=0.5,
            iv_rank=95.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )
        modifier_min = validator.get_confidence_modifier(data_min)
        assert modifier_min >= 0.8

    def test_custom_thresholds(self) -> None:
        """Test validator with custom thresholds."""
        custom_validator = OptionsValidator(
            volume_spike_ratio=3.0,
            iv_rank_warning_threshold=70.0,
        )

        # Volume ratio 2.5 should NOT trigger with 3.0 threshold
        data = OptionsFlowData(
            volume_ratio=2.5,
            iv_rank=50.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )
        is_enhanced, _ = custom_validator.validate(data)
        assert is_enhanced is False

        # IV rank 75 SHOULD trigger warning with 70.0 threshold
        data_iv = OptionsFlowData(
            volume_ratio=1.0,
            iv_rank=75.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )
        _, warnings = custom_validator.validate(data_iv)
        assert len(warnings) == 1

    def test_edge_case_exact_threshold(self, validator: OptionsValidator) -> None:
        """Test behavior at exact threshold values."""
        # Volume ratio exactly at threshold
        data = OptionsFlowData(
            volume_ratio=2.0,
            iv_rank=50.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )
        is_enhanced, _ = validator.validate(data)
        assert is_enhanced is True  # >= threshold

        # IV rank exactly at threshold
        data_iv = OptionsFlowData(
            volume_ratio=1.0,
            iv_rank=80.0,
            put_call_ratio=0.8,
            unusual_activity=False,
        )
        _, warnings = validator.validate(data_iv)
        assert len(warnings) == 0  # > threshold, not >=
