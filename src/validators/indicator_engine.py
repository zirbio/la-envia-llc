"""IndicatorEngine for calculating technical indicators from price data."""

import pandas as pd
import pandas_ta as ta


class IndicatorEngine:
    """Calculates technical indicators from price data."""

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index).

        The RSI is a momentum oscillator that measures the speed and magnitude
        of directional price movements. It ranges from 0 to 100, with values
        above 70 typically indicating overbought conditions and values below
        30 indicating oversold conditions.

        Args:
            prices: Series of closing prices
            period: RSI period (default 14)

        Returns:
            Current RSI value (0-100), or 50.0 if insufficient data

        Raises:
            None - handles all edge cases gracefully
        """
        # Handle edge cases
        if prices is None or len(prices) == 0:
            return 50.0

        # Check for all NaN values
        if prices.isna().all():
            return 50.0

        # Check if we have enough data for the calculation
        # RSI needs at least period+1 values to calculate properly
        if len(prices) < period + 1:
            return 50.0

        # Calculate RSI using pandas_ta
        rsi_series = ta.rsi(prices, length=period)

        # Get the last (most recent) RSI value
        if rsi_series is None or len(rsi_series) == 0:
            return 50.0

        last_rsi = rsi_series.iloc[-1]

        # Handle NaN result (can happen with flat prices or edge cases)
        if pd.isna(last_rsi):
            return 50.0

        # Ensure RSI is within valid range [0, 100]
        # This is defensive programming - pandas_ta should already ensure this
        rsi_value = float(last_rsi)
        rsi_value = max(0.0, min(100.0, rsi_value))

        return rsi_value

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, str]:
        """
        Calculate MACD histogram and trend.

        The MACD (Moving Average Convergence Divergence) is a trend-following
        momentum indicator that shows the relationship between two moving averages
        of a security's price. The histogram represents the difference between
        the MACD line and the signal line.

        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            Tuple of (histogram_value, trend_direction)
            trend_direction is "rising", "falling", or "flat"

        Raises:
            None - handles all edge cases gracefully
        """
        # Handle edge cases
        if prices is None or len(prices) == 0:
            return (0.0, "flat")

        # Check for all NaN values
        if prices.isna().all():
            return (0.0, "flat")

        # MACD needs at least slow + signal periods for proper calculation
        # Typically this means at least 26 + 9 = 35 data points
        min_required = slow + signal
        if len(prices) < min_required:
            return (0.0, "flat")

        # Calculate MACD using pandas_ta
        # Returns DataFrame with columns:
        # MACD_fast_slow_signal, MACDh_fast_slow_signal, MACDs_fast_slow_signal
        macd_df = ta.macd(prices, fast=fast, slow=slow, signal=signal)

        # Handle None or empty result
        if macd_df is None or len(macd_df) == 0:
            return (0.0, "flat")

        # Get the histogram column name (MACDh_12_26_9 for default parameters)
        histogram_col = f"MACDh_{fast}_{slow}_{signal}"

        # Check if the column exists
        if histogram_col not in macd_df.columns:
            return (0.0, "flat")

        # Get histogram values
        histogram_series = macd_df[histogram_col]

        # Get the last (most recent) histogram value
        if histogram_series is None or len(histogram_series) == 0:
            return (0.0, "flat")

        last_histogram = histogram_series.iloc[-1]

        # Handle NaN result
        if pd.isna(last_histogram):
            return (0.0, "flat")

        histogram_value = float(last_histogram)

        # Determine trend by comparing last 3 histogram values
        # Need at least 3 values to determine trend
        if len(histogram_series) < 3:
            return (histogram_value, "flat")

        # Get last 3 histogram values
        last_3_values = histogram_series.iloc[-3:].tolist()

        # Filter out NaN values
        valid_values = [v for v in last_3_values if not pd.isna(v)]

        # Need at least 2 valid values to compare
        if len(valid_values) < 2:
            return (histogram_value, "flat")

        # Determine trend
        # Rising: each value is greater than the previous
        # Falling: each value is less than the previous
        # Flat: otherwise
        if len(valid_values) >= 3:
            if valid_values[2] > valid_values[1] > valid_values[0]:
                trend = "rising"
            elif valid_values[2] < valid_values[1] < valid_values[0]:
                trend = "falling"
            else:
                trend = "flat"
        elif len(valid_values) == 2:
            # Only 2 values: simple comparison
            if valid_values[1] > valid_values[0]:
                trend = "rising"
            elif valid_values[1] < valid_values[0]:
                trend = "falling"
            else:
                trend = "flat"
        else:
            trend = "flat"

        return (histogram_value, trend)
