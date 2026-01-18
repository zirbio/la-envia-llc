# src/research/data_fetchers/market_fetcher.py

"""Fetcher for market data (futures, VIX, pre-market gappers)."""

import logging
import yfinance as yf

from src.research.data_fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)


class MarketFetcher(BaseFetcher):
    """Fetches futures, VIX, and pre-market gapper data."""

    def __init__(
        self,
        min_gap_percent: float = 3.0,
        min_volume: int = 100000,
    ):
        """Initialize the market fetcher.

        Args:
            min_gap_percent: Minimum gap percentage for gappers.
            min_volume: Minimum pre-market volume for gappers.
        """
        super().__init__("market")
        self.min_gap_percent = min_gap_percent
        self.min_volume = min_volume

    async def fetch(self) -> dict:
        """Fetch market data.

        Returns:
            Dictionary with futures, VIX, and gapper data.
        """
        result = {
            "futures": {"es": None, "nq": None, "es_change": None, "nq_change": None},
            "vix": None,
            "gappers": [],
        }

        try:
            # Fetch futures
            es = yf.Ticker("ES=F")
            nq = yf.Ticker("NQ=F")
            vix = yf.Ticker("^VIX")

            es_info = es.info
            nq_info = nq.info
            vix_info = vix.info

            result["futures"] = {
                "es": es_info.get("regularMarketPrice"),
                "nq": nq_info.get("regularMarketPrice"),
                "es_change": es_info.get("regularMarketChangePercent"),
                "nq_change": nq_info.get("regularMarketChangePercent"),
            }

            result["vix"] = vix_info.get("regularMarketPrice")

            # Fetch gappers (simplified - would use screener in production)
            result["gappers"] = await self._fetch_gappers()

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            result["error"] = str(e)

        return result

    async def _fetch_gappers(self) -> list[dict]:
        """Fetch pre-market gappers.

        Returns:
            List of gapper dictionaries.
        """
        # In production, this would scrape Finviz or use a screener API
        # For now, return empty list - will be enhanced later
        return []
