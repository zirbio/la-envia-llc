# tests/scoring/test_dynamic_weight_calculator.py
"""Tests for dynamic weight calculator."""
import pytest

from src.scoring.dynamic_weight_calculator import DynamicWeightCalculator


class TestDynamicWeightCalculator:
    """Tests for DynamicWeightCalculator."""

    def test_strong_trend_favors_technical(self):
        """Test ADX > 30 should return (0.4, 0.6) - favoring technical."""
        calculator = DynamicWeightCalculator()

        sentiment_weight, technical_weight = calculator.calculate_weights(adx=35.0)

        assert sentiment_weight == 0.4
        assert technical_weight == 0.6

    def test_weak_trend_favors_sentiment(self):
        """Test ADX < 20 should return (0.6, 0.4) - favoring sentiment."""
        calculator = DynamicWeightCalculator()

        sentiment_weight, technical_weight = calculator.calculate_weights(adx=15.0)

        assert sentiment_weight == 0.6
        assert technical_weight == 0.4

    def test_normal_trend_balanced_weights(self):
        """Test ADX 20-30 should return (0.5, 0.5) - balanced weights."""
        calculator = DynamicWeightCalculator()

        # Test lower boundary
        sentiment_weight, technical_weight = calculator.calculate_weights(adx=20.0)
        assert sentiment_weight == 0.5
        assert technical_weight == 0.5

        # Test upper boundary
        sentiment_weight, technical_weight = calculator.calculate_weights(adx=30.0)
        assert sentiment_weight == 0.5
        assert technical_weight == 0.5

        # Test middle value
        sentiment_weight, technical_weight = calculator.calculate_weights(adx=25.0)
        assert sentiment_weight == 0.5
        assert technical_weight == 0.5

    def test_weights_always_sum_to_one(self):
        """Test that weights always sum to 1.0 for various scenarios."""
        calculator = DynamicWeightCalculator()

        # Strong trend
        sentiment, technical = calculator.calculate_weights(adx=40.0)
        assert sentiment + technical == pytest.approx(1.0)

        # Weak trend
        sentiment, technical = calculator.calculate_weights(adx=10.0)
        assert sentiment + technical == pytest.approx(1.0)

        # Normal trend
        sentiment, technical = calculator.calculate_weights(adx=25.0)
        assert sentiment + technical == pytest.approx(1.0)

        # High volatility override
        sentiment, technical = calculator.calculate_weights(
            adx=25.0, volatility_percentile=85.0
        )
        assert sentiment + technical == pytest.approx(1.0)

        # Edge cases
        sentiment, technical = calculator.calculate_weights(adx=0.0)
        assert sentiment + technical == pytest.approx(1.0)

        sentiment, technical = calculator.calculate_weights(adx=100.0)
        assert sentiment + technical == pytest.approx(1.0)

    def test_high_volatility_overrides_adx(self):
        """Test volatility > 80 should return (0.35, 0.65) regardless of ADX."""
        calculator = DynamicWeightCalculator()

        # High volatility with strong trend ADX
        sentiment_weight, technical_weight = calculator.calculate_weights(
            adx=35.0, volatility_percentile=85.0
        )
        assert sentiment_weight == 0.35
        assert technical_weight == 0.65

        # High volatility with weak trend ADX
        sentiment_weight, technical_weight = calculator.calculate_weights(
            adx=15.0, volatility_percentile=90.0
        )
        assert sentiment_weight == 0.35
        assert technical_weight == 0.65

        # High volatility with normal trend ADX
        sentiment_weight, technical_weight = calculator.calculate_weights(
            adx=25.0, volatility_percentile=81.0
        )
        assert sentiment_weight == 0.35
        assert technical_weight == 0.65

    def test_volatility_at_boundary_does_not_override(self):
        """Test volatility at exactly 80 does NOT trigger override."""
        calculator = DynamicWeightCalculator()

        # At 80%, should NOT override - use ADX-based weights
        sentiment_weight, technical_weight = calculator.calculate_weights(
            adx=35.0, volatility_percentile=80.0
        )
        # Should be strong trend weights (0.4, 0.6), not high volatility (0.35, 0.65)
        assert sentiment_weight == 0.4
        assert technical_weight == 0.6

    def test_custom_strong_trend_threshold(self):
        """Test custom strong_trend_adx threshold."""
        calculator = DynamicWeightCalculator(strong_trend_adx=25.0)

        # ADX 27 is now strong trend (> 25)
        sentiment_weight, technical_weight = calculator.calculate_weights(adx=27.0)
        assert sentiment_weight == 0.4
        assert technical_weight == 0.6

    def test_custom_weak_trend_threshold(self):
        """Test custom weak_trend_adx threshold."""
        calculator = DynamicWeightCalculator(weak_trend_adx=25.0)

        # ADX 23 is now weak trend (< 25)
        sentiment_weight, technical_weight = calculator.calculate_weights(adx=23.0)
        assert sentiment_weight == 0.6
        assert technical_weight == 0.4

    def test_custom_strong_trend_technical_weight(self):
        """Test custom strong_trend_technical_weight."""
        calculator = DynamicWeightCalculator(strong_trend_technical_weight=0.7)

        sentiment_weight, technical_weight = calculator.calculate_weights(adx=35.0)
        assert sentiment_weight == pytest.approx(0.3)  # 1 - 0.7
        assert technical_weight == 0.7

    def test_custom_weak_trend_sentiment_weight(self):
        """Test custom weak_trend_sentiment_weight."""
        calculator = DynamicWeightCalculator(weak_trend_sentiment_weight=0.7)

        sentiment_weight, technical_weight = calculator.calculate_weights(adx=15.0)
        assert sentiment_weight == 0.7
        assert technical_weight == pytest.approx(0.3)  # 1 - 0.7

    def test_custom_base_weights(self):
        """Test custom base weights for normal trend."""
        calculator = DynamicWeightCalculator(
            base_sentiment_weight=0.55,
            base_technical_weight=0.45,
        )

        sentiment_weight, technical_weight = calculator.calculate_weights(adx=25.0)
        assert sentiment_weight == 0.55
        assert technical_weight == 0.45

    def test_default_volatility_percentile(self):
        """Test default volatility_percentile is 50.0 (no override)."""
        calculator = DynamicWeightCalculator()

        # Default volatility percentile should not trigger override
        sentiment_weight, technical_weight = calculator.calculate_weights(adx=25.0)
        assert sentiment_weight == 0.5
        assert technical_weight == 0.5
