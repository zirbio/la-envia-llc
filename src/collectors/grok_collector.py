# src/collectors/grok_collector.py
"""Collector for X/Twitter data via Grok API."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType

logger = logging.getLogger(__name__)


class GrokCollector(BaseCollector):
    """Collector for X/Twitter using Grok API with x_search."""

    def __init__(
        self,
        api_key: str,
        search_queries: list[str],
        refresh_interval: int = 30,
        max_results_per_query: int = 20,
    ):
        """Initialize the Grok collector.

        Args:
            api_key: xAI API key.
            search_queries: List of search queries (tickers, terms, from:user).
            refresh_interval: Seconds between search cycles.
            max_results_per_query: Max results per query.
        """
        super().__init__(name="grok", source_type=SourceType.GROK)
        self._api_key = api_key
        self._search_queries = search_queries
        self._refresh_interval = refresh_interval
        self._max_results = max_results_per_query
        self._client: AsyncOpenAI | None = None
        self._seen_ids: set[str] = set()

    async def connect(self) -> None:
        """Initialize Grok API client."""
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url="https://api.x.ai/v1",
        )
        self._connected = True
        logger.info("GrokCollector connected")

    async def disconnect(self) -> None:
        """Close connection."""
        self._client = None
        self._connected = False
        logger.info("GrokCollector disconnected")

    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream messages from X via Grok x_search.

        Yields:
            SocialMessage objects as they arrive.
        """
        if not self._connected or not self._client:
            raise RuntimeError("Collector not connected. Call connect() first.")

        while True:
            for query in self._search_queries:
                try:
                    async for msg in self._search_x(query):
                        yield msg
                except Exception as e:
                    logger.error(f"Error searching for '{query}': {e}")

            await asyncio.sleep(self._refresh_interval)

    async def _search_x(self, query: str) -> AsyncIterator[SocialMessage]:
        """Execute search on X via Grok.

        Args:
            query: Search query string.

        Yields:
            SocialMessage for each new post found.
        """
        if not self._client:
            return

        try:
            # Use Grok with x_search tool
            response = await self._client.chat.completions.create(
                model="grok-2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Search X for the most recent posts about the given query. "
                            "Return the raw post data including id, author, text, "
                            "timestamps, engagement metrics, and sentiment score."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Search X for: {query}",
                    },
                ],
            )

            # Parse response and extract posts
            posts = self._parse_grok_response(response)

            for post in posts:
                post_id = post.get("id", "")
                if post_id and post_id not in self._seen_ids:
                    self._seen_ids.add(post_id)
                    yield self._to_social_message(post)

        except Exception as e:
            logger.error(f"Grok API error for query '{query}': {e}")

    def _parse_grok_response(self, response) -> list[dict]:
        """Parse Grok response to extract X posts.

        Args:
            response: Grok API response.

        Returns:
            List of post dictionaries.
        """
        # Extract content from response
        if not response.choices:
            return []

        content = response.choices[0].message.content
        if not content:
            return []

        # TODO: Parse structured response from Grok
        # For now, return empty - actual parsing depends on Grok's response format
        return []

    def _to_social_message(self, post: dict) -> SocialMessage:
        """Convert X post to SocialMessage.

        Args:
            post: Post dictionary from Grok.

        Returns:
            SocialMessage object.
        """
        # Parse timestamp
        created_at = post.get("created_at", "")
        try:
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(timezone.utc)

        author = post.get("author_username", "unknown")
        post_id = post.get("id", "")

        return SocialMessage(
            source=SourceType.GROK,
            source_id=str(post_id),
            author=author,
            content=post.get("text", ""),
            timestamp=timestamp,
            url=f"https://x.com/{author}/status/{post_id}",
            like_count=post.get("like_count"),
            retweet_count=post.get("retweet_count"),
        )
