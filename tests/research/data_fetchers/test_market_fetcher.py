# tests/research/data_fetchers/test_market_fetcher.py

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.research.data_fetchers.market_fetcher import MarketFetcher


class TestMarketFetcher:
    def test_inherits_base_fetcher(self):
        from src.research.data_fetchers.base import BaseFetcher
        fetcher = MarketFetcher()
        assert isinstance(fetcher, BaseFetcher)
        assert fetcher.name == "market"

    @pytest.mark.asyncio
    async def test_fetch_returns_expected_structure(self):
        fetcher = MarketFetcher()

        # Mock yfinance
        with patch("src.research.data_fetchers.market_fetcher.yf") as mock_yf:
            # Mock futures data
            mock_es = MagicMock()
            mock_es.info = {"regularMarketPrice": 5000.0, "regularMarketChangePercent": 0.5}

            mock_nq = MagicMock()
            mock_nq.info = {"regularMarketPrice": 17500.0, "regularMarketChangePercent": 0.8}

            mock_vix = MagicMock()
            mock_vix.info = {"regularMarketPrice": 14.5}

            mock_yf.Ticker.side_effect = lambda x: {
                "ES=F": mock_es,
                "NQ=F": mock_nq,
                "^VIX": mock_vix,
            }.get(x, MagicMock())

            result = await fetcher.fetch()

        assert "futures" in result
        assert "vix" in result
        assert "gappers" in result

    @pytest.mark.asyncio
    async def test_fetch_handles_errors_gracefully(self):
        fetcher = MarketFetcher()

        with patch("src.research.data_fetchers.market_fetcher.yf") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("API Error")

            result = await fetcher.fetch()

        # Should return empty/default structure, not raise
        assert "futures" in result
        assert "error" in result or result["futures"]["es"] is None
