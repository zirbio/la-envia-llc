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

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[float, float]:
        """
        Calculate Stochastic %K and %D.

        The Stochastic Oscillator is a momentum indicator that compares a
        particular closing price to a range of prices over a certain period.
        %K is the fast stochastic, and %D is the slow stochastic (SMA of %K).

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            k_period: Period for %K calculation (default 14)
            d_period: Period for %D calculation (default 3)

        Returns:
            Tuple of (stochastic_k, stochastic_d), both in range 0-100
            Returns (50.0, 50.0) if insufficient data

        Raises:
            None - handles all edge cases gracefully
        """
        # Handle edge cases
        if (
            high is None
            or low is None
            or close is None
            or len(high) == 0
            or len(low) == 0
            or len(close) == 0
        ):
            return (50.0, 50.0)

        # Check for all NaN values
        if high.isna().all() or low.isna().all() or close.isna().all():
            return (50.0, 50.0)

        # Check if series lengths match
        if not (len(high) == len(low) == len(close)):
            return (50.0, 50.0)

        # Stochastic needs at least k_period + d_period for proper calculation
        min_required = k_period + d_period
        if len(close) < min_required:
            return (50.0, 50.0)

        # Calculate Stochastic using pandas_ta
        # Returns DataFrame with columns like STOCHk_14_3_3 and STOCHd_14_3_3
        stoch_df = ta.stoch(high, low, close, k=k_period, d=d_period)

        # Handle None or empty result
        if stoch_df is None or len(stoch_df) == 0:
            return (50.0, 50.0)

        # Get the column names (format: STOCHk_{k}_{d}_3, STOCHd_{k}_{d}_3)
        k_col = f"STOCHk_{k_period}_{d_period}_3"
        d_col = f"STOCHd_{k_period}_{d_period}_3"

        # Check if columns exist
        if k_col not in stoch_df.columns or d_col not in stoch_df.columns:
            return (50.0, 50.0)

        # Get the last (most recent) values
        k_series = stoch_df[k_col]
        d_series = stoch_df[d_col]

        if k_series is None or len(k_series) == 0:
            return (50.0, 50.0)
        if d_series is None or len(d_series) == 0:
            return (50.0, 50.0)

        last_k = k_series.iloc[-1]
        last_d = d_series.iloc[-1]

        # Handle NaN results
        if pd.isna(last_k) or pd.isna(last_d):
            return (50.0, 50.0)

        # Convert to float and ensure within valid range [0, 100]
        k_value = float(last_k)
        d_value = float(last_d)

        k_value = max(0.0, min(100.0, k_value))
        d_value = max(0.0, min(100.0, d_value))

        return (k_value, d_value)

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float:
        """
        Calculate ADX (Average Directional Index).

        The ADX is a trend strength indicator that quantifies the strength
        of a trend regardless of its direction. Values range from 0 to 100,
        with values above 25 typically indicating a strong trend.

        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: ADX period (default 14)

        Returns:
            ADX value (0-100), or 0.0 if insufficient data

        Raises:
            None - handles all edge cases gracefully
        """
        # Handle edge cases
        if (
            high is None
            or low is None
            or close is None
            or len(high) == 0
            or len(low) == 0
            or len(close) == 0
        ):
            return 0.0

        # Check for all NaN values
        if high.isna().all() or low.isna().all() or close.isna().all():
            return 0.0

        # Check if series lengths match
        if not (len(high) == len(low) == len(close)):
            return 0.0

        # ADX needs at least 2 * period for proper calculation
        # (period for DI calculation, period for ADX smoothing)
        min_required = 2 * period
        if len(close) < min_required:
            return 0.0

        # Calculate ADX using pandas_ta
        # Returns DataFrame with columns like ADX_{period}, DMP_{period}, DMN_{period}
        adx_df = ta.adx(high, low, close, length=period)

        # Handle None or empty result
        if adx_df is None or len(adx_df) == 0:
            return 0.0

        # Get the ADX column name (format: ADX_{period})
        adx_col = f"ADX_{period}"

        # Check if column exists
        if adx_col not in adx_df.columns:
            return 0.0

        # Get the last (most recent) ADX value
        adx_series = adx_df[adx_col]

        if adx_series is None or len(adx_series) == 0:
            return 0.0

        last_adx = adx_series.iloc[-1]

        # Handle NaN result
        if pd.isna(last_adx):
            return 0.0

        # Convert to float and ensure within valid range [0, 100]
        adx_value = float(last_adx)
        adx_value = max(0.0, min(100.0, adx_value))

        return adx_value
