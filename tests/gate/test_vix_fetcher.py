"""Tests for VIX fetcher."""

from unittest.mock import MagicMock, patch

import pytest

from gate.vix_fetcher import VixFetcher


class TestVixFetcher:
    """Tests for VixFetcher class."""

    def test_fetch_vix_returns_value(self) -> None:
        """fetch_vix returns current VIX value from yfinance."""
        fetcher = VixFetcher()

        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 18.5}

        with patch("gate.vix_fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetcher.fetch_vix()

        assert result == 18.5

    def test_fetch_vix_uses_previous_close_as_fallback(self) -> None:
        """fetch_vix uses previousClose if regularMarketPrice not available."""
        fetcher = VixFetcher()

        mock_ticker = MagicMock()
        mock_ticker.info = {"previousClose": 20.0}

        with patch("gate.vix_fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetcher.fetch_vix()

        assert result == 20.0

    def test_fetch_vix_returns_none_on_error(self) -> None:
        """fetch_vix returns None when yfinance fails."""
        fetcher = VixFetcher()

        with patch("gate.vix_fetcher.yf.Ticker", side_effect=Exception("Network error")):
            result = fetcher.fetch_vix()

        assert result is None

    def test_fetch_vix_returns_none_when_no_price_data(self) -> None:
        """fetch_vix returns None when no price data available."""
        fetcher = VixFetcher()

        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("gate.vix_fetcher.yf.Ticker", return_value=mock_ticker):
            result = fetcher.fetch_vix()

        assert result is None
