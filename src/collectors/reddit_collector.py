# src/collectors/reddit_collector.py
from datetime import datetime, timezone
from typing import AsyncIterator

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType


class RedditCollector(BaseCollector):
    """Collector for Reddit using asyncpraw."""

    def __init__(
        self,
        subreddits: list[str],
        client_id: str,
        client_secret: str,
        user_agent: str,
        use_streaming: bool = True,
    ):
        super().__init__(name="reddit", source_type=SourceType.REDDIT)
        self._subreddits = subreddits
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._use_streaming = use_streaming
        self._reddit = None

    @property
    def subreddits(self) -> list[str]:
        return self._subreddits

    async def connect(self) -> None:
        """Initialize asyncpraw Reddit connection."""
        try:
            import asyncpraw
            self._reddit = asyncpraw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            self._connected = True
        except ImportError:
            raise RuntimeError("asyncpraw not installed. Run: pip install asyncpraw")

    async def disconnect(self) -> None:
        """Close Reddit connection."""
        if self._reddit:
            await self._reddit.close()
        self._reddit = None
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream submissions from subreddits."""
        if not self._connected or self._reddit is None:
            raise RuntimeError("Collector not connected. Call connect() first.")

        subreddit_str = "+".join(self._subreddits)
        subreddit = await self._reddit.subreddit(subreddit_str)

        if self._use_streaming:
            async for submission in subreddit.stream.submissions(skip_existing=True):
                yield self._parse_submission(submission)
        else:
            async for submission in subreddit.new(limit=25):
                yield self._parse_submission(submission)

    def _parse_submission(self, submission) -> SocialMessage:
        """Convert a Reddit submission to SocialMessage."""
        content = submission.title
        if submission.selftext:
            content = f"{submission.title}\n\n{submission.selftext}"

        author_name = "[deleted]"
        if submission.author:
            author_name = submission.author.name

        return SocialMessage(
            source=SourceType.REDDIT,
            source_id=submission.id,
            author=author_name,
            content=content,
            timestamp=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
            url=f"https://reddit.com{submission.permalink}",
            subreddit=submission.subreddit.display_name,
            upvotes=submission.score,
            comment_count=submission.num_comments,
        )
