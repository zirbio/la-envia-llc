# tests/collectors/test_stocktwits_collector.py
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from src.collectors.stocktwits_collector import StocktwitsCollector
from src.models.social_message import SourceType


class TestStocktwitsCollector:
    @pytest.fixture
    def collector(self):
        return StocktwitsCollector(watchlist=["AAPL", "NVDA", "TSLA"])

    def test_collector_has_correct_source_type(self, collector):
        assert collector.source_type == SourceType.STOCKTWITS

    def test_collector_has_correct_name(self, collector):
        assert collector.name == "stocktwits"

    def test_collector_stores_watchlist(self, collector):
        assert "AAPL" in collector.watchlist
        assert "NVDA" in collector.watchlist
        assert "TSLA" in collector.watchlist

    def test_collector_has_default_refresh_interval(self, collector):
        assert collector._refresh_interval == 30

    def test_collector_accepts_custom_refresh_interval(self):
        collector = StocktwitsCollector(watchlist=["AAPL"], refresh_interval=60)
        assert collector._refresh_interval == 60

    def test_parse_message_to_social_message(self, collector):
        mock_msg = {
            "id": 12345,
            "body": "$NVDA looking bullish today!",
            "created_at": "2026-01-17T10:30:00Z",
            "user": {"username": "stocktrader99"},
            "entities": {
                "sentiment": {"basic": "Bullish"}
            },
        }

        msg = collector._parse_message(mock_msg, ticker="NVDA")

        assert msg.source == SourceType.STOCKTWITS
        assert msg.source_id == "12345"
        assert msg.author == "stocktrader99"
        assert "$NVDA" in msg.content
        assert msg.sentiment == "Bullish"
        assert "stocktwits.com" in msg.url

    def test_parse_message_no_sentiment(self, collector):
        mock_msg = {
            "id": 67890,
            "body": "What do you think about $AAPL?",
            "created_at": "2026-01-17T11:00:00Z",
            "user": {"username": "newbie"},
            "entities": {},
        }

        msg = collector._parse_message(mock_msg, ticker="AAPL")
        assert msg.sentiment is None

    def test_parse_message_bearish_sentiment(self, collector):
        mock_msg = {
            "id": 11111,
            "body": "$TSLA looks weak",
            "created_at": "2026-01-17T12:00:00Z",
            "user": {"username": "bearish_trader"},
            "entities": {
                "sentiment": {"basic": "Bearish"}
            },
        }

        msg = collector._parse_message(mock_msg, ticker="TSLA")
        assert msg.sentiment == "Bearish"

    def test_parse_message_handles_missing_entities(self, collector):
        mock_msg = {
            "id": 22222,
            "body": "Testing $SPY",
            "created_at": "2026-01-17T13:00:00Z",
            "user": {"username": "tester"},
        }

        msg = collector._parse_message(mock_msg, ticker="SPY")
        assert msg.sentiment is None

    def test_parse_message_handles_invalid_timestamp(self, collector):
        mock_msg = {
            "id": 33333,
            "body": "Test message",
            "created_at": "invalid-date",
            "user": {"username": "user"},
            "entities": {},
        }

        msg = collector._parse_message(mock_msg, ticker="TEST")
        # Should use current time as fallback
        assert msg.timestamp is not None
        assert isinstance(msg.timestamp, datetime)

    def test_parse_message_handles_missing_user(self, collector):
        mock_msg = {
            "id": 44444,
            "body": "Anonymous message",
            "created_at": "2026-01-17T14:00:00Z",
            "entities": {},
        }

        msg = collector._parse_message(mock_msg, ticker="ANON")
        assert msg.author == "unknown"

    def test_extract_tickers_from_stocktwits_message(self, collector):
        mock_msg = {
            "id": 55555,
            "body": "$AAPL and $MSFT looking good",
            "created_at": "2026-01-17T15:00:00Z",
            "user": {"username": "analyst"},
            "entities": {},
        }

        msg = collector._parse_message(mock_msg, ticker="AAPL")
        tickers = msg.extract_tickers()

        assert "AAPL" in tickers
        assert "MSFT" in tickers

    @pytest.mark.asyncio
    async def test_connect_sets_connected_flag(self, collector):
        with patch.dict("sys.modules", {"pytwits": MagicMock()}):
            await collector.connect()
            assert collector.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_raises_when_pytwits_not_installed(self, collector):
        with patch.dict("sys.modules", {"pytwits": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'pytwits'")):
                with pytest.raises(RuntimeError, match="pytwits not installed"):
                    await collector.connect()

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self, collector):
        collector._connected = True
        collector._client = MagicMock()

        await collector.disconnect()

        assert collector.is_connected is False
        assert collector._client is None

    @pytest.mark.asyncio
    async def test_stream_raises_when_not_connected(self, collector):
        with pytest.raises(RuntimeError, match="not connected"):
            async for _ in collector.stream():
                pass

    @pytest.mark.asyncio
    async def test_stream_yields_messages(self, collector):
        # Setup mock client
        mock_client = MagicMock()
        mock_client.get_symbol_msgs.return_value = {
            "messages": [
                {
                    "id": 1,
                    "body": "$AAPL up 5%",
                    "created_at": "2026-01-17T16:00:00Z",
                    "user": {"username": "trader1"},
                    "entities": {"sentiment": {"basic": "Bullish"}},
                },
                {
                    "id": 2,
                    "body": "$AAPL earnings tomorrow",
                    "created_at": "2026-01-17T16:01:00Z",
                    "user": {"username": "trader2"},
                    "entities": {},
                },
            ]
        }

        collector._connected = True
        collector._client = mock_client

        messages = []
        async for msg in collector.stream():
            messages.append(msg)
            if len(messages) >= 2:
                break

        assert len(messages) >= 2
        assert all(msg.source == SourceType.STOCKTWITS for msg in messages)
