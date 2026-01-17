"""Tests for IndicatorEngine with RSI calculation."""

import numpy as np
import pandas as pd
import pytest

from src.validators.indicator_engine import IndicatorEngine


class TestIndicatorEngine:
    """Test suite for IndicatorEngine class."""

    @pytest.fixture
    def engine(self) -> IndicatorEngine:
        """Create an IndicatorEngine instance for testing."""
        return IndicatorEngine()

    def test_rsi_uptrend_above_50(self, engine: IndicatorEngine) -> None:
        """Test that RSI is above 50 for uptrend prices."""
        # Create uptrend data: steadily increasing prices
        prices = pd.Series([100 + i for i in range(30)])

        rsi = engine.calculate_rsi(prices, period=14)

        assert rsi > 50.0, "RSI should be above 50 for uptrend"
        assert 0 <= rsi <= 100, "RSI should be between 0 and 100"

    def test_rsi_downtrend_below_50(self, engine: IndicatorEngine) -> None:
        """Test that RSI is below 50 for downtrend prices."""
        # Create downtrend data: steadily decreasing prices
        prices = pd.Series([100 - i for i in range(30)])

        rsi = engine.calculate_rsi(prices, period=14)

        assert rsi < 50.0, "RSI should be below 50 for downtrend"
        assert 0 <= rsi <= 100, "RSI should be between 0 and 100"

    def test_rsi_range(self, engine: IndicatorEngine) -> None:
        """Test that RSI is always between 0 and 100."""
        # Test with various price patterns
        test_cases = [
            # Volatile uptrend
            pd.Series([100, 105, 103, 108, 106, 112, 110, 115, 113, 120,
                      118, 125, 123, 130, 128, 135, 133, 140, 138, 145]),
            # Volatile downtrend
            pd.Series([100, 95, 97, 92, 94, 88, 90, 85, 87, 80,
                      82, 75, 77, 70, 72, 65, 67, 60, 62, 55]),
            # Sideways/choppy market
            pd.Series([100, 102, 99, 101, 98, 103, 97, 104, 96, 105,
                      95, 106, 94, 107, 93, 108, 92, 109, 91, 110]),
            # Extreme volatility
            pd.Series([100, 150, 80, 140, 90, 130, 70, 160, 60, 170,
                      50, 180, 40, 190, 30, 200, 20, 210, 10, 220]),
        ]

        for prices in test_cases:
            rsi = engine.calculate_rsi(prices, period=14)
            assert 0 <= rsi <= 100, f"RSI {rsi} is outside valid range [0, 100]"

    def test_rsi_insufficient_data(self, engine: IndicatorEngine) -> None:
        """Test that RSI returns 50.0 (neutral) when data is insufficient."""
        # Test with data shorter than period
        short_prices = pd.Series([100, 101, 102])
        rsi = engine.calculate_rsi(short_prices, period=14)
        assert rsi == 50.0, "Should return 50.0 for insufficient data"

        # Test with empty series
        empty_prices = pd.Series([], dtype=float)
        rsi = engine.calculate_rsi(empty_prices, period=14)
        assert rsi == 50.0, "Should return 50.0 for empty series"

        # Test with series of NaN values
        nan_prices = pd.Series([np.nan, np.nan, np.nan])
        rsi = engine.calculate_rsi(nan_prices, period=14)
        assert rsi == 50.0, "Should return 50.0 for NaN values"

    def test_rsi_with_different_periods(self, engine: IndicatorEngine) -> None:
        """Test RSI calculation with different period values."""
        prices = pd.Series([100 + i for i in range(50)])

        # Test with period=7 (shorter)
        rsi_7 = engine.calculate_rsi(prices, period=7)
        assert 0 <= rsi_7 <= 100, "RSI should be in valid range for period=7"

        # Test with period=21 (longer)
        rsi_21 = engine.calculate_rsi(prices, period=21)
        assert 0 <= rsi_21 <= 100, "RSI should be in valid range for period=21"

        # Test with default period (14)
        rsi_default = engine.calculate_rsi(prices)
        assert 0 <= rsi_default <= 100, (
            "RSI should be in valid range for default period"
        )

    def test_rsi_neutral_market(self, engine: IndicatorEngine) -> None:
        """Test RSI is near 50 for neutral/sideways market."""
        # Create perfectly flat prices
        flat_prices = pd.Series([100.0] * 30)
        rsi = engine.calculate_rsi(flat_prices, period=14)

        # With flat prices, RSI calculation might return NaN or default to 50
        # We accept either 50.0 or a value very close to 50
        assert rsi == 50.0 or abs(rsi - 50.0) < 0.1, \
            f"RSI should be 50.0 or very close for flat prices, got {rsi}"

    def test_macd_histogram_positive_uptrend(self, engine: IndicatorEngine) -> None:
        """Test that MACD histogram is positive for uptrend prices."""
        # Create exponential uptrend data (accelerating growth)
        # This creates a realistic uptrend where MACD histogram is positive
        prices = pd.Series([100 * (1.02 ** i) for i in range(50)])

        histogram, trend = engine.calculate_macd(prices, fast=12, slow=26, signal=9)

        assert histogram > 0, "MACD histogram should be positive for uptrend"
        assert trend in ["rising", "falling", "flat"], (
            f"Trend should be 'rising', 'falling', or 'flat', got {trend}"
        )

    def test_macd_trend_rising(self, engine: IndicatorEngine) -> None:
        """Test MACD trend detection for rising, falling, and flat."""
        # Test rising trend: exponential growth (accelerating momentum)
        rising_prices = pd.Series([100 * (1.02 ** i) for i in range(50)])
        _, trend_rising = engine.calculate_macd(rising_prices)
        assert trend_rising == "rising", "Should detect rising trend"

        # Test falling trend: exponential decay (accelerating downtrend)
        falling_prices = pd.Series([200 * (0.98 ** i) for i in range(50)])
        _, trend_falling = engine.calculate_macd(falling_prices)
        assert trend_falling == "falling", "Should detect falling trend"

        # Test flat trend: sideways movement
        flat_prices = pd.Series([100.0 + (i % 3 - 1) * 0.5 for i in range(50)])
        _, trend_flat = engine.calculate_macd(flat_prices)
        assert trend_flat in ["rising", "falling", "flat"], (
            "Should return valid trend for flat market"
        )

    def test_macd_insufficient_data(self, engine: IndicatorEngine) -> None:
        """Test that MACD returns (0.0, 'flat') for insufficient data."""
        # Test with very short data
        short_prices = pd.Series([100, 101, 102])
        histogram, trend = engine.calculate_macd(short_prices)
        assert histogram == 0.0, "Should return 0.0 histogram for insufficient data"
        assert trend == "flat", "Should return 'flat' trend for insufficient data"

        # Test with empty series
        empty_prices = pd.Series([], dtype=float)
        histogram, trend = engine.calculate_macd(empty_prices)
        assert histogram == 0.0, "Should return 0.0 histogram for empty series"
        assert trend == "flat", "Should return 'flat' trend for empty series"

        # Test with NaN values
        nan_prices = pd.Series([np.nan, np.nan, np.nan])
        histogram, trend = engine.calculate_macd(nan_prices)
        assert histogram == 0.0, "Should return 0.0 histogram for NaN values"
        assert trend == "flat", "Should return 'flat' trend for NaN values"

    def test_stochastic_range(self, engine: IndicatorEngine) -> None:
        """Test that Stochastic %K and %D are always between 0 and 100."""
        # Create test data with various price patterns
        high = pd.Series([105, 110, 108, 112, 115, 120, 118, 125, 122, 130,
                         128, 135, 133, 140, 138, 145, 143, 150, 148, 155])
        low = pd.Series([95, 98, 96, 100, 103, 108, 106, 112, 110, 118,
                        115, 123, 120, 128, 125, 133, 130, 138, 135, 143])
        close = pd.Series([100, 105, 103, 108, 110, 115, 113, 120, 118, 125,
                          123, 130, 128, 135, 133, 140, 138, 145, 143, 150])

        stoch_k, stoch_d = engine.calculate_stochastic(
            high, low, close, k_period=14, d_period=3
        )

        assert 0 <= stoch_k <= 100, f"Stochastic %K {stoch_k} is outside [0, 100]"
        assert 0 <= stoch_d <= 100, f"Stochastic %D {stoch_d} is outside [0, 100]"

    def test_stochastic_insufficient_data(self, engine: IndicatorEngine) -> None:
        """Test that Stochastic returns (50.0, 50.0) for insufficient data."""
        # Test with very short data
        high = pd.Series([105, 110, 108])
        low = pd.Series([95, 98, 96])
        close = pd.Series([100, 105, 103])

        stoch_k, stoch_d = engine.calculate_stochastic(high, low, close)
        assert stoch_k == 50.0, "Should return 50.0 for %K with insufficient data"
        assert stoch_d == 50.0, "Should return 50.0 for %D with insufficient data"

        # Test with empty series
        empty = pd.Series([], dtype=float)
        stoch_k, stoch_d = engine.calculate_stochastic(empty, empty, empty)
        assert stoch_k == 50.0, "Should return 50.0 for %K with empty series"
        assert stoch_d == 50.0, "Should return 50.0 for %D with empty series"

        # Test with NaN values
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        stoch_k, stoch_d = engine.calculate_stochastic(
            nan_series, nan_series, nan_series
        )
        assert stoch_k == 50.0, "Should return 50.0 for %K with NaN values"
        assert stoch_d == 50.0, "Should return 50.0 for %D with NaN values"

    def test_adx_range(self, engine: IndicatorEngine) -> None:
        """Test that ADX is always between 0 and 100."""
        # Create trending data (strong uptrend should have higher ADX)
        high = pd.Series([105 + i*2 for i in range(30)])
        low = pd.Series([95 + i*2 for i in range(30)])
        close = pd.Series([100 + i*2 for i in range(30)])

        adx = engine.calculate_adx(high, low, close, period=14)

        assert 0 <= adx <= 100, f"ADX {adx} is outside [0, 100]"

    def test_adx_insufficient_data(self, engine: IndicatorEngine) -> None:
        """Test that ADX returns 0.0 for insufficient data."""
        # Test with very short data
        high = pd.Series([105, 110, 108])
        low = pd.Series([95, 98, 96])
        close = pd.Series([100, 105, 103])

        adx = engine.calculate_adx(high, low, close)
        assert adx == 0.0, "Should return 0.0 for ADX with insufficient data"

        # Test with empty series
        empty = pd.Series([], dtype=float)
        adx = engine.calculate_adx(empty, empty, empty)
        assert adx == 0.0, "Should return 0.0 for ADX with empty series"

        # Test with NaN values
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        adx = engine.calculate_adx(nan_series, nan_series, nan_series)
        assert adx == 0.0, "Should return 0.0 for ADX with NaN values"

    def test_calculate_all_returns_indicators(self, engine: IndicatorEngine) -> None:
        """Test that calculate_all returns TechnicalIndicators with valid values."""
        # Create OHLC DataFrame with sufficient data for all indicators
        # Need at least 35 data points for MACD (26 + 9)
        ohlc = pd.DataFrame({
            'open': [100 + i * 0.5 for i in range(50)],
            'high': [105 + i * 0.5 for i in range(50)],
            'low': [95 + i * 0.5 for i in range(50)],
            'close': [100 + i for i in range(50)]
        })

        result = engine.calculate_all(ohlc)

        # Check that result is a TechnicalIndicators object
        from src.validators.models import TechnicalIndicators
        assert isinstance(result, TechnicalIndicators), (
            "Should return TechnicalIndicators instance"
        )

        # Check RSI is in valid range
        assert 0 <= result.rsi <= 100, f"RSI {result.rsi} is outside [0, 100]"

        # Check MACD values
        assert isinstance(result.macd_histogram, float), (
            "MACD histogram should be a float"
        )
        assert result.macd_trend in ["rising", "falling", "flat"], (
            f"MACD trend should be 'rising', 'falling', or 'flat', got {result.macd_trend}"
        )

        # Check Stochastic values
        assert 0 <= result.stochastic_k <= 100, (
            f"Stochastic %K {result.stochastic_k} is outside [0, 100]"
        )
        assert 0 <= result.stochastic_d <= 100, (
            f"Stochastic %D {result.stochastic_d} is outside [0, 100]"
        )

        # Check ADX is in valid range
        assert 0 <= result.adx <= 100, f"ADX {result.adx} is outside [0, 100]"

        # Check divergence fields are None (placeholder)
        assert result.rsi_divergence is None, (
            "RSI divergence should be None for now"
        )
        assert result.macd_divergence is None, (
            "MACD divergence should be None for now"
        )

    def test_calculate_all_with_insufficient_data(self, engine: IndicatorEngine) -> None:
        """Test that calculate_all returns neutral defaults with insufficient data."""
        # Create OHLC DataFrame with insufficient data (only 5 rows)
        ohlc = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [100, 101, 102, 103, 104]
        })

        result = engine.calculate_all(ohlc)

        # Check that result is a TechnicalIndicators object
        from src.validators.models import TechnicalIndicators
        assert isinstance(result, TechnicalIndicators), (
            "Should return TechnicalIndicators instance"
        )

        # Check RSI returns neutral default (50.0)
        assert result.rsi == 50.0, (
            f"RSI should be 50.0 for insufficient data, got {result.rsi}"
        )

        # Check MACD returns neutral defaults (0.0, 'flat')
        assert result.macd_histogram == 0.0, (
            f"MACD histogram should be 0.0 for insufficient data, got {result.macd_histogram}"
        )
        assert result.macd_trend == "flat", (
            f"MACD trend should be 'flat' for insufficient data, got {result.macd_trend}"
        )

        # Check Stochastic returns neutral defaults (50.0, 50.0)
        assert result.stochastic_k == 50.0, (
            f"Stochastic %K should be 50.0 for insufficient data, got {result.stochastic_k}"
        )
        assert result.stochastic_d == 50.0, (
            f"Stochastic %D should be 50.0 for insufficient data, got {result.stochastic_d}"
        )

        # Check ADX returns neutral default (0.0)
        assert result.adx == 0.0, (
            f"ADX should be 0.0 for insufficient data, got {result.adx}"
        )

        # Check divergence fields are None
        assert result.rsi_divergence is None, (
            "RSI divergence should be None"
        )
        assert result.macd_divergence is None, (
            "MACD divergence should be None"
        )
