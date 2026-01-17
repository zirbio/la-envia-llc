# tests/validation/test_mock_market_data.py
import pytest
import pandas as pd
from src.validation.mock_market_data import MockMarketData, MarketTrend


class TestMockMarketData:
    def test_generate_ohlcv_returns_dataframe(self):
        mock = MockMarketData()
        df = mock.generate_ohlcv("AAPL", bars=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_generate_uptrend(self):
        mock = MockMarketData(trend=MarketTrend.UPTREND)
        df = mock.generate_ohlcv("AAPL", bars=50)
        assert df["close"].iloc[-1] > df["close"].iloc[0]

    def test_generate_downtrend(self):
        mock = MockMarketData(trend=MarketTrend.DOWNTREND)
        df = mock.generate_ohlcv("AAPL", bars=50)
        assert df["close"].iloc[-1] < df["close"].iloc[0]

    def test_generate_sideways(self):
        mock = MockMarketData(trend=MarketTrend.SIDEWAYS)
        df = mock.generate_ohlcv("AAPL", bars=50)
        pct_change = abs(df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
        assert pct_change < 0.05

    def test_high_always_above_low(self):
        mock = MockMarketData()
        df = mock.generate_ohlcv("AAPL", bars=100)
        assert (df["high"] >= df["low"]).all()

    def test_open_close_within_high_low(self):
        mock = MockMarketData()
        df = mock.generate_ohlcv("AAPL", bars=100)
        assert (df["open"] <= df["high"]).all()
        assert (df["open"] >= df["low"]).all()
        assert (df["close"] <= df["high"]).all()
        assert (df["close"] >= df["low"]).all()
