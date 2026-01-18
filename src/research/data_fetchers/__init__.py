"""Data fetchers for Morning Research Agent."""

from src.research.data_fetchers.base import BaseFetcher
from src.research.data_fetchers.market_fetcher import MarketFetcher

__all__ = ["BaseFetcher", "MarketFetcher"]
