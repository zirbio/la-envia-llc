# tests/collectors/test_base.py
import pytest
from datetime import datetime, timezone
from typing import AsyncIterator
from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType


class MockCollector(BaseCollector):
    """Mock collector for testing base class."""

    def __init__(self):
        super().__init__(name="mock", source_type=SourceType.TWITTER)
        self._messages = []

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        for msg in self._messages:
            yield msg

    def add_message(self, msg: SocialMessage):
        self._messages.append(msg)


class TestBaseCollector:
    @pytest.fixture
    def collector(self):
        return MockCollector()

    @pytest.mark.asyncio
    async def test_collector_starts_disconnected(self, collector):
        assert not collector.is_connected

    @pytest.mark.asyncio
    async def test_collector_connects(self, collector):
        await collector.connect()
        assert collector.is_connected

    @pytest.mark.asyncio
    async def test_collector_disconnects(self, collector):
        await collector.connect()
        await collector.disconnect()
        assert not collector.is_connected

    @pytest.mark.asyncio
    async def test_collector_streams_messages(self, collector):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="test",
            content="$AAPL looking good",
            timestamp=datetime.now(timezone.utc),
        )
        collector.add_message(msg)

        await collector.connect()
        messages = [m async for m in collector.stream()]
        assert len(messages) == 1
        assert messages[0].author == "test"

    def test_collector_has_name(self, collector):
        assert collector.name == "mock"

    def test_collector_has_source_type(self, collector):
        assert collector.source_type == SourceType.TWITTER
