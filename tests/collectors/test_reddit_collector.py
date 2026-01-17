# tests/collectors/test_reddit_collector.py
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from src.collectors.reddit_collector import RedditCollector
from src.models.social_message import SourceType


class TestRedditCollector:
    @pytest.fixture
    def collector(self):
        subreddits = ["wallstreetbets", "stocks"]
        return RedditCollector(
            subreddits=subreddits,
            client_id="test_id",
            client_secret="test_secret",
            user_agent="TestBot/1.0",
        )

    def test_collector_has_correct_source_type(self, collector):
        assert collector.source_type == SourceType.REDDIT

    def test_collector_has_correct_name(self, collector):
        assert collector.name == "reddit"

    def test_collector_stores_subreddits(self, collector):
        assert "wallstreetbets" in collector.subreddits
        assert "stocks" in collector.subreddits

    def test_parse_submission_to_message(self, collector):
        mock_submission = MagicMock()
        mock_submission.id = "abc123"
        mock_submission.author.name = "trader99"
        mock_submission.title = "DD on $AAPL earnings"
        mock_submission.selftext = "Here's my analysis..."
        mock_submission.created_utc = datetime.now(timezone.utc).timestamp()
        mock_submission.permalink = "/r/stocks/comments/abc123"
        mock_submission.subreddit.display_name = "stocks"
        mock_submission.score = 150
        mock_submission.num_comments = 45

        msg = collector._parse_submission(mock_submission)

        assert msg.source == SourceType.REDDIT
        assert msg.source_id == "abc123"
        assert msg.author == "trader99"
        assert "$AAPL" in msg.content
        assert msg.subreddit == "stocks"
        assert msg.upvotes == 150
        assert msg.comment_count == 45

    def test_parse_submission_combines_title_and_body(self, collector):
        mock_submission = MagicMock()
        mock_submission.id = "xyz789"
        mock_submission.author.name = "user"
        mock_submission.title = "$NVDA breaking out"
        mock_submission.selftext = "Great volume on $NVDA today"
        mock_submission.created_utc = datetime.now(timezone.utc).timestamp()
        mock_submission.permalink = "/r/stocks/xyz789"
        mock_submission.subreddit.display_name = "stocks"
        mock_submission.score = 10
        mock_submission.num_comments = 5

        msg = collector._parse_submission(mock_submission)

        assert "$NVDA breaking out" in msg.content
        assert "Great volume" in msg.content

    def test_extract_tickers_from_reddit_post(self, collector):
        mock_submission = MagicMock()
        mock_submission.id = "test"
        mock_submission.author.name = "user"
        mock_submission.title = "$GME $AMC to the moon"
        mock_submission.selftext = "Also looking at $BB"
        mock_submission.created_utc = datetime.now(timezone.utc).timestamp()
        mock_submission.permalink = "/r/wsb/test"
        mock_submission.subreddit.display_name = "wallstreetbets"
        mock_submission.score = 1000
        mock_submission.num_comments = 500

        msg = collector._parse_submission(mock_submission)
        tickers = msg.extract_tickers()

        assert "GME" in tickers
        assert "AMC" in tickers
        assert "BB" in tickers

    def test_parse_submission_handles_deleted_author(self, collector):
        mock_submission = MagicMock()
        mock_submission.id = "deleted123"
        mock_submission.author = None  # Deleted authors are None
        mock_submission.title = "Some post"
        mock_submission.selftext = ""
        mock_submission.created_utc = datetime.now(timezone.utc).timestamp()
        mock_submission.permalink = "/r/stocks/deleted123"
        mock_submission.subreddit.display_name = "stocks"
        mock_submission.score = 5
        mock_submission.num_comments = 2

        msg = collector._parse_submission(mock_submission)

        assert msg.author == "[deleted]"

    @pytest.mark.asyncio
    async def test_connect_sets_connected_flag(self, collector):
        with patch.dict("sys.modules", {"asyncpraw": MagicMock()}):
            await collector.connect()
            assert collector._connected is True

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self, collector):
        mock_reddit = AsyncMock()
        collector._connected = True
        collector._reddit = mock_reddit

        await collector.disconnect()

        assert collector._connected is False
        assert collector._reddit is None
        mock_reddit.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_raises_when_not_connected(self, collector):
        with pytest.raises(RuntimeError, match="Collector not connected"):
            async for _ in collector.stream():
                pass
