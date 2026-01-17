# src/collectors/stocktwits_collector.py
from datetime import datetime, timezone
from typing import AsyncIterator

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType


class StocktwitsCollector(BaseCollector):
    """Collector for Stocktwits using pytwits."""

    def __init__(
        self,
        watchlist: list[str],
        refresh_interval: int = 30,
    ):
        super().__init__(name="stocktwits", source_type=SourceType.STOCKTWITS)
        self._watchlist = watchlist
        self._refresh_interval = refresh_interval
        self._client = None
        self._last_message_ids: dict[str, int] = {}

    @property
    def watchlist(self) -> list[str]:
        return self._watchlist

    async def connect(self) -> None:
        """Initialize pytwits client."""
        try:
            from pytwits import Streamer
            self._client = Streamer()
            self._connected = True
        except ImportError:
            raise RuntimeError("pytwits not installed. Run: pip install pytwits")

    async def disconnect(self) -> None:
        """Close client."""
        self._client = None
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream messages for watchlist tickers."""
        if not self._connected or self._client is None:
            raise RuntimeError("Collector not connected. Call connect() first.")

        for ticker in self._watchlist:
            async for msg in self._get_ticker_messages(ticker):
                yield msg

    async def _get_ticker_messages(self, ticker: str, limit: int = 30):
        """Get recent messages for a ticker."""
        if self._client is None:
            return

        try:
            response = self._client.get_symbol_msgs(ticker)
            messages = response.get("messages", [])

            for msg in messages[:limit]:
                msg_id = msg.get("id", 0)
                last_id = self._last_message_ids.get(ticker, 0)

                if msg_id > last_id:
                    self._last_message_ids[ticker] = msg_id
                    yield self._parse_message(msg, ticker)
        except Exception as e:
            print(f"Error fetching Stocktwits messages for {ticker}: {e}")

    def _parse_message(self, msg: dict, ticker: str) -> SocialMessage:
        """Convert a Stocktwits message to SocialMessage."""
        created_at = msg.get("created_at", "")
        try:
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(timezone.utc)

        sentiment = None
        entities = msg.get("entities", {})
        if entities:
            sentiment_data = entities.get("sentiment", {})
            sentiment = sentiment_data.get("basic")

        return SocialMessage(
            source=SourceType.STOCKTWITS,
            source_id=str(msg.get("id", "")),
            author=msg.get("user", {}).get("username", "unknown"),
            content=msg.get("body", ""),
            timestamp=timestamp,
            url=f"https://stocktwits.com/symbol/{ticker}",
            sentiment=sentiment,
        )
