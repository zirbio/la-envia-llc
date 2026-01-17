# src/validation/mock_market_data.py
from enum import Enum
import pandas as pd
import numpy as np


class MarketTrend(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class MockMarketData:
    """Generates mock OHLCV data for testing."""

    def __init__(
        self,
        trend: MarketTrend = MarketTrend.SIDEWAYS,
        base_price: float = 100.0,
        volatility: float = 0.02,
    ):
        self.trend = trend
        self.base_price = base_price
        self.volatility = volatility

    def generate_ohlcv(self, symbol: str, bars: int = 50) -> pd.DataFrame:
        """Generate OHLCV DataFrame for testing."""
        np.random.seed(42)  # Reproducible for tests

        prices = [self.base_price]
        for _ in range(bars - 1):
            change = np.random.normal(0, self.volatility)
            if self.trend == MarketTrend.UPTREND:
                change += 0.005  # Stronger upward drift
            elif self.trend == MarketTrend.DOWNTREND:
                change -= 0.005  # Stronger downward drift
            elif self.trend == MarketTrend.SIDEWAYS:
                change *= 0.3  # Significantly reduce volatility for sideways
            elif self.trend == MarketTrend.VOLATILE:
                change *= 2
            prices.append(prices[-1] * (1 + change))

        data = []
        for price in prices:
            noise = np.random.uniform(0.005, 0.015)
            high = price * (1 + noise)
            low = price * (1 - noise)
            open_price = np.random.uniform(low, high)
            close_price = np.random.uniform(low, high)
            volume = int(np.random.uniform(100000, 1000000))
            data.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume,
            })

        return pd.DataFrame(data)
