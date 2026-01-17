# tests/collectors/test_twitter_collector.py
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from src.collectors.twitter_collector import TwitterCollector
from src.models.social_message import SourceType


class TestTwitterCollector:
    @pytest.fixture
    def collector(self):
        accounts = ["unusual_whales", "FirstSquawk"]
        return TwitterCollector(accounts_to_follow=accounts)

    def test_collector_has_correct_source_type(self, collector):
        assert collector.source_type == SourceType.TWITTER

    def test_collector_has_correct_name(self, collector):
        assert collector.name == "twitter"

    def test_collector_stores_accounts(self, collector):
        assert "unusual_whales" in collector.accounts_to_follow
        assert "FirstSquawk" in collector.accounts_to_follow

    @pytest.mark.asyncio
    async def test_parse_tweet_to_message(self, collector):
        mock_tweet = MagicMock()
        mock_tweet.id = 123456789
        mock_tweet.user.username = "unusual_whales"
        mock_tweet.rawContent = "Large $NVDA call sweep $2.4M"
        mock_tweet.date = datetime.now(timezone.utc)
        mock_tweet.url = "https://twitter.com/unusual_whales/status/123456789"
        mock_tweet.retweetCount = 100
        mock_tweet.likeCount = 500

        msg = collector._parse_tweet(mock_tweet)

        assert msg.source == SourceType.TWITTER
        assert msg.source_id == "123456789"
        assert msg.author == "unusual_whales"
        assert "$NVDA" in msg.content
        assert msg.retweet_count == 100
        assert msg.like_count == 500

    @pytest.mark.asyncio
    async def test_extract_tickers_from_parsed_message(self, collector):
        mock_tweet = MagicMock()
        mock_tweet.id = 123
        mock_tweet.user.username = "test"
        mock_tweet.rawContent = "Looking at $AAPL $MSFT today"
        mock_tweet.date = datetime.now(timezone.utc)
        mock_tweet.url = "https://twitter.com/test/123"
        mock_tweet.retweetCount = 0
        mock_tweet.likeCount = 0

        msg = collector._parse_tweet(mock_tweet)
        tickers = msg.extract_tickers()

        assert "AAPL" in tickers
        assert "MSFT" in tickers

    @pytest.mark.asyncio
    async def test_connect_sets_connected_flag(self, collector):
        with patch.dict("sys.modules", {"twscrape": MagicMock()}):
            await collector.connect()
            assert collector._connected is True

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self, collector):
        collector._connected = True
        collector._api = MagicMock()

        await collector.disconnect()

        assert collector._connected is False
        assert collector._api is None

    @pytest.mark.asyncio
    async def test_stream_raises_when_not_connected(self, collector):
        with pytest.raises(RuntimeError, match="Collector not connected"):
            async for _ in collector.stream():
                pass
