# src/collectors/twitter_collector.py
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType


class TwitterCollector(BaseCollector):
    """Collector for Twitter/X using twscrape."""

    def __init__(
        self,
        accounts_to_follow: list[str],
        refresh_interval: int = 15,
    ):
        super().__init__(name="twitter", source_type=SourceType.TWITTER)
        self._accounts_to_follow = accounts_to_follow
        self._refresh_interval = refresh_interval
        self._api = None
        self._last_tweet_ids: dict[str, int] = {}

    @property
    def accounts_to_follow(self) -> list[str]:
        return self._accounts_to_follow

    async def connect(self) -> None:
        """Initialize twscrape API connection."""
        try:
            from twscrape import API
            self._api = API()
            self._connected = True
        except ImportError:
            raise RuntimeError("twscrape not installed. Run: pip install twscrape")

    async def disconnect(self) -> None:
        """Close connection."""
        self._api = None
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream tweets from followed accounts."""
        if not self._connected or self._api is None:
            raise RuntimeError("Collector not connected. Call connect() first.")

        for username in self._accounts_to_follow:
            async for tweet in self._get_user_tweets(username):
                yield self._parse_tweet(tweet)

    async def _get_user_tweets(self, username: str, limit: int = 10):
        """Get recent tweets from a user."""
        if self._api is None:
            return

        try:
            user = await self._api.user_by_login(username)
            if user:
                async for tweet in self._api.user_tweets(user.id, limit=limit):
                    last_id = self._last_tweet_ids.get(username, 0)
                    if tweet.id > last_id:
                        self._last_tweet_ids[username] = tweet.id
                        yield tweet
        except Exception:
            pass

    def _parse_tweet(self, tweet) -> SocialMessage:
        """Convert a twscrape Tweet to SocialMessage."""
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id=str(tweet.id),
            author=tweet.user.username,
            content=tweet.rawContent,
            timestamp=tweet.date if tweet.date.tzinfo else tweet.date.replace(tzinfo=timezone.utc),
            url=tweet.url,
            retweet_count=tweet.retweetCount,
            like_count=tweet.likeCount,
        )
