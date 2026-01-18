# tests/collectors/test_grok_collector.py
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_grok_collector_init():
    from src.collectors.grok_collector import GrokCollector
    from src.models.social_message import SourceType

    collector = GrokCollector(
        api_key="test-key",
        search_queries=["$NVDA", "$TSLA"],
        refresh_interval=30,
    )

    assert collector.name == "grok"
    assert collector.source_type == SourceType.GROK
    assert not collector.is_connected


@pytest.mark.asyncio
async def test_grok_collector_connect():
    from src.collectors.grok_collector import GrokCollector

    with patch("src.collectors.grok_collector.AsyncOpenAI") as mock_client:
        collector = GrokCollector(
            api_key="test-key",
            search_queries=["$NVDA"],
        )

        await collector.connect()

        assert collector.is_connected
        mock_client.assert_called_once()


@pytest.mark.asyncio
async def test_grok_collector_disconnect():
    from src.collectors.grok_collector import GrokCollector

    with patch("src.collectors.grok_collector.AsyncOpenAI"):
        collector = GrokCollector(
            api_key="test-key",
            search_queries=["$NVDA"],
        )

        await collector.connect()
        assert collector.is_connected

        await collector.disconnect()
        assert not collector.is_connected


def test_parse_grok_response():
    from src.collectors.grok_collector import GrokCollector
    from src.models.social_message import SourceType

    collector = GrokCollector(
        api_key="test-key",
        search_queries=["$NVDA"],
    )

    # Simulate Grok response with X posts
    mock_post = {
        "id": "123456789",
        "author_username": "unusual_whales",
        "text": "$NVDA massive call flow detected",
        "created_at": "2026-01-18T10:30:00Z",
        "like_count": 500,
        "retweet_count": 100,
        "sentiment": 0.8,  # Grok native sentiment
    }

    message = collector._to_social_message(mock_post)

    assert message.source == SourceType.GROK
    assert message.source_id == "123456789"
    assert message.author == "unusual_whales"
    assert "$NVDA" in message.content
    assert message.like_count == 500
