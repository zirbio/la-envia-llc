# tests/clients/test_alpaca_market_data.py
"""Tests for AlpacaClient market data operations."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.execution.alpaca_client import AlpacaClient


class TestAlpacaClientMarketData:
    """Tests for market data retrieval methods."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

    def test_get_bars_returns_dataframe(self, client):
        """get_bars should return a pandas DataFrame with OHLCV data."""
        # Create mock bar data
        mock_bar = MagicMock()
        mock_bar.open = 140.50
        mock_bar.high = 142.00
        mock_bar.low = 139.75
        mock_bar.close = 141.25
        mock_bar.volume = 1000000

        # Create mock response with MultiIndex DataFrame structure
        mock_bars_df = pd.DataFrame(
            {
                "open": [140.50],
                "high": [142.00],
                "low": [139.75],
                "close": [141.25],
                "volume": [1000000],
            },
            index=pd.MultiIndex.from_tuples(
                [("NVDA", pd.Timestamp("2024-01-17 09:30:00"))],
                names=["symbol", "timestamp"],
            ),
        )

        # Mock the data client
        client._data_client = MagicMock()
        client._data_client.get_stock_bars.return_value = mock_bars_df

        # Call get_bars
        result = client.get_bars(symbol="NVDA", limit=50, timeframe="5Min")

        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
        assert result["open"].iloc[0] == 140.50
        assert result["close"].iloc[0] == 141.25

    def test_get_bars_with_symbol(self, client):
        """get_bars should request data for the correct symbol."""
        # Create mock DataFrame
        mock_bars_df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.5],
                "close": [101.5, 102.5],
                "volume": [500000, 600000],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("AAPL", pd.Timestamp("2024-01-17 09:30:00")),
                    ("AAPL", pd.Timestamp("2024-01-17 09:35:00")),
                ],
                names=["symbol", "timestamp"],
            ),
        )

        # Mock the data client
        client._data_client = MagicMock()
        client._data_client.get_stock_bars.return_value = mock_bars_df

        # Call get_bars with specific symbol
        result = client.get_bars(symbol="AAPL", limit=2, timeframe="5Min")

        # Verify the data client was called
        client._data_client.get_stock_bars.assert_called_once()

        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_get_bars_with_different_timeframes(self, client):
        """get_bars should handle different timeframe strings."""
        # Test various timeframes
        timeframes = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

        for tf in timeframes:
            # Create mock DataFrame
            mock_bars_df = pd.DataFrame(
                {
                    "open": [140.50],
                    "high": [142.00],
                    "low": [139.75],
                    "close": [141.25],
                    "volume": [1000000],
                },
                index=pd.MultiIndex.from_tuples(
                    [("NVDA", pd.Timestamp("2024-01-17 09:30:00"))],
                    names=["symbol", "timestamp"],
                ),
            )

            # Mock the data client
            client._data_client = MagicMock()
            client._data_client.get_stock_bars.return_value = mock_bars_df

            # Call get_bars
            result = client.get_bars(symbol="NVDA", limit=1, timeframe=tf)

            # Verify it returns a DataFrame
            assert isinstance(result, pd.DataFrame)

    def test_get_bars_returns_empty_dataframe_when_no_data(self, client):
        """get_bars should return empty DataFrame when no data available."""
        # Mock empty DataFrame response
        empty_df = pd.DataFrame()

        client._data_client = MagicMock()
        client._data_client.get_stock_bars.return_value = empty_df

        result = client.get_bars(symbol="INVALID", limit=50, timeframe="5Min")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_bars_handles_multiindex_dataframe(self, client):
        """get_bars should handle MultiIndex DataFrame and droplevel if needed."""
        # Create MultiIndex DataFrame (typical Alpaca response)
        mock_bars_df = pd.DataFrame(
            {
                "open": [140.50, 141.00],
                "high": [142.00, 142.50],
                "low": [139.75, 140.00],
                "close": [141.25, 141.75],
                "volume": [1000000, 1100000],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("NVDA", pd.Timestamp("2024-01-17 09:30:00")),
                    ("NVDA", pd.Timestamp("2024-01-17 09:35:00")),
                ],
                names=["symbol", "timestamp"],
            ),
        )

        client._data_client = MagicMock()
        client._data_client.get_stock_bars.return_value = mock_bars_df

        result = client.get_bars(symbol="NVDA", limit=2, timeframe="5Min")

        # Result should have timestamp as index (symbol level dropped)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @patch("src.execution.alpaca_client.StockBarsRequest")
    @patch("src.execution.alpaca_client.TimeFrame")
    @patch("src.execution.alpaca_client.TimeFrameUnit")
    def test_get_bars_creates_correct_request(
        self, mock_timeframe_unit, mock_timeframe, mock_bars_request, client
    ):
        """get_bars should create StockBarsRequest with correct parameters."""
        # Mock the TimeFrameUnit enum
        mock_timeframe_unit.Minute = "Minute"
        mock_timeframe_unit.Hour = "Hour"
        mock_timeframe_unit.Day = "Day"

        # Create mock DataFrame
        mock_bars_df = pd.DataFrame(
            {
                "open": [140.50],
                "high": [142.00],
                "low": [139.75],
                "close": [141.25],
                "volume": [1000000],
            },
            index=pd.MultiIndex.from_tuples(
                [("NVDA", pd.Timestamp("2024-01-17 09:30:00"))],
                names=["symbol", "timestamp"],
            ),
        )

        client._data_client = MagicMock()
        client._data_client.get_stock_bars.return_value = mock_bars_df

        # Call get_bars
        result = client.get_bars(symbol="NVDA", limit=50, timeframe="5Min")

        # Verify StockBarsRequest was called
        mock_bars_request.assert_called_once()

        # Verify result
        assert isinstance(result, pd.DataFrame)
