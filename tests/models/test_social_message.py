# tests/models/test_social_message.py
import pytest
from datetime import datetime, timezone
from src.models.social_message import SocialMessage, SourceType


class TestSocialMessage:
    def test_create_twitter_message(self):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="tweet_123",
            author="unusual_whales",
            content="🚨 Large $NVDA call sweep",
            timestamp=datetime.now(timezone.utc),
            url="https://twitter.com/unusual_whales/status/123",
        )
        assert msg.source == SourceType.TWITTER
        assert msg.author == "unusual_whales"
        assert "$NVDA" in msg.content

    def test_create_reddit_message(self):
        msg = SocialMessage(
            source=SourceType.REDDIT,
            source_id="post_abc",
            author="trader123",
            content="DD on $AAPL earnings play",
            timestamp=datetime.now(timezone.utc),
            url="https://reddit.com/r/stocks/abc",
            subreddit="stocks",
        )
        assert msg.source == SourceType.REDDIT
        assert msg.subreddit == "stocks"

    def test_extract_tickers_from_content(self):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="test",
            content="Looking at $AAPL $MSFT and $GOOGL today",
            timestamp=datetime.now(timezone.utc),
        )
        tickers = msg.extract_tickers()
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_extract_tickers_no_duplicates(self):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="test",
            content="$NVDA calls, $NVDA puts, all $NVDA",
            timestamp=datetime.now(timezone.utc),
        )
        tickers = msg.extract_tickers()
        assert tickers == ["NVDA"]

    def test_extract_tickers_excludes_crypto(self):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="test",
            content="$BTC $ETH $AAPL $DOGE",
            timestamp=datetime.now(timezone.utc),
        )
        tickers = msg.extract_tickers(exclude_crypto=True)
        assert tickers == ["AAPL"]
