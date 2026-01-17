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
