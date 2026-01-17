# tests/collectors/test_collector_manager.py
import pytest
import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

from src.collectors.collector_manager import CollectorManager
from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType


class MockCollector(BaseCollector):
    """Mock collector for testing purposes."""

    def __init__(self, name: str, messages: list[SocialMessage]):
        super().__init__(name=name, source_type=SourceType.TWITTER)
        self._messages = messages

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        for msg in self._messages:
            yield msg


class SlowMockCollector(BaseCollector):
    """Mock collector that yields messages with delays."""

    def __init__(self, name: str, messages: list[SocialMessage], delay: float = 0.1):
        super().__init__(name=name, source_type=SourceType.TWITTER)
        self._messages = messages
        self._delay = delay

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        for msg in self._messages:
            await asyncio.sleep(self._delay)
            yield msg


class TestCollectorManager:
    @pytest.fixture
    def messages(self):
        return [
            SocialMessage(
                source=SourceType.TWITTER,
                source_id="1",
                author="user1",
                content="$AAPL bullish",
                timestamp=datetime.now(timezone.utc),
            ),
            SocialMessage(
                source=SourceType.REDDIT,
                source_id="2",
                author="user2",
                content="$NVDA DD",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

    @pytest.fixture
    def manager(self, messages):
        collector1 = MockCollector("twitter", [messages[0]])
        collector2 = MockCollector("reddit", [messages[1]])
        return CollectorManager(collectors=[collector1, collector2])

    def test_manager_has_collectors(self, manager):
        """Test that manager stores collectors properly."""
        assert len(manager.collectors) == 2

    def test_manager_collectors_property_returns_list(self, manager):
        """Test that collectors property returns a list."""
        assert isinstance(manager.collectors, list)
        assert all(isinstance(c, BaseCollector) for c in manager.collectors)

    def test_manager_callbacks_initially_empty(self, manager):
        """Test that callbacks list is empty initially."""
        assert manager.callbacks == []

    def test_add_callback(self, manager):
        """Test adding a callback function."""
        callback = MagicMock()
        manager.add_callback(callback)
        assert callback in manager.callbacks

    def test_add_multiple_callbacks(self, manager):
        """Test adding multiple callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        manager.add_callback(callback1)
        manager.add_callback(callback2)
        assert len(manager.callbacks) == 2
        assert callback1 in manager.callbacks
        assert callback2 in manager.callbacks

    @pytest.mark.asyncio
    async def test_manager_connects_all(self, manager):
        """Test that connect_all connects all collectors."""
        await manager.connect_all()
        for collector in manager.collectors:
            assert collector.is_connected

    @pytest.mark.asyncio
    async def test_manager_disconnects_all(self, manager):
        """Test that disconnect_all disconnects all collectors."""
        await manager.connect_all()
        await manager.disconnect_all()
        for collector in manager.collectors:
            assert not collector.is_connected

    @pytest.mark.asyncio
    async def test_manager_collects_from_all(self, manager, messages):
        """Test that stream_all collects messages from all collectors."""
        await manager.connect_all()

        collected = []
        async for msg in manager.stream_all():
            collected.append(msg)
            if len(collected) >= 2:
                break

        assert len(collected) == 2
        authors = {m.author for m in collected}
        assert "user1" in authors
        assert "user2" in authors

    @pytest.mark.asyncio
    async def test_stream_all_yields_social_messages(self, manager):
        """Test that stream_all yields SocialMessage instances."""
        await manager.connect_all()

        async for msg in manager.stream_all():
            assert isinstance(msg, SocialMessage)
            break

    @pytest.mark.asyncio
    async def test_callbacks_called_for_each_message(self, manager):
        """Test that callbacks are called for each streamed message."""
        callback = MagicMock()
        manager.add_callback(callback)
        await manager.connect_all()

        collected = []
        async for msg in manager.stream_all():
            collected.append(msg)
            if len(collected) >= 2:
                break

        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_callback_receives_message(self, manager, messages):
        """Test that callback receives the actual message."""
        received_messages = []
        callback = lambda msg: received_messages.append(msg)
        manager.add_callback(callback)
        await manager.connect_all()

        async for msg in manager.stream_all():
            if len(received_messages) >= 2:
                break

        assert len(received_messages) == 2

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_stop_stream(self):
        """Test that a failing callback doesn't stop the stream."""
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="1",
            author="user1",
            content="test",
            timestamp=datetime.now(timezone.utc),
        )
        collector = MockCollector("twitter", [msg])
        manager = CollectorManager(collectors=[collector])

        def failing_callback(m):
            raise ValueError("Callback error")

        manager.add_callback(failing_callback)
        await manager.connect_all()

        collected = []
        async for m in manager.stream_all():
            collected.append(m)

        assert len(collected) == 1

    @pytest.mark.asyncio
    async def test_empty_collectors_list(self):
        """Test manager with empty collectors list."""
        manager = CollectorManager(collectors=[])
        assert len(manager.collectors) == 0
        await manager.connect_all()
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_stream_all_from_empty_manager(self):
        """Test streaming from manager with no collectors."""
        manager = CollectorManager(collectors=[])
        await manager.connect_all()

        collected = []
        async for msg in manager.stream_all():
            collected.append(msg)

        assert collected == []

    @pytest.mark.asyncio
    async def test_run_method_connects_and_disconnects(self):
        """Test that run() connects, streams, and disconnects."""
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="1",
            author="user1",
            content="test",
            timestamp=datetime.now(timezone.utc),
        )
        collector = MockCollector("twitter", [msg])
        manager = CollectorManager(collectors=[collector])

        received = []
        manager.add_callback(lambda m: received.append(m))

        await manager.run()

        # After run completes, collectors should be disconnected
        assert not collector.is_connected
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_concurrent_streams(self):
        """Test that messages from multiple slow collectors interleave."""
        msg1 = SocialMessage(
            source=SourceType.TWITTER,
            source_id="1",
            author="slow1",
            content="$AAPL",
            timestamp=datetime.now(timezone.utc),
        )
        msg2 = SocialMessage(
            source=SourceType.REDDIT,
            source_id="2",
            author="slow2",
            content="$NVDA",
            timestamp=datetime.now(timezone.utc),
        )

        collector1 = SlowMockCollector("slow1", [msg1], delay=0.05)
        collector2 = SlowMockCollector("slow2", [msg2], delay=0.05)
        manager = CollectorManager(collectors=[collector1, collector2])

        await manager.connect_all()

        collected = []
        async for msg in manager.stream_all():
            collected.append(msg)
            if len(collected) >= 2:
                break

        assert len(collected) == 2
