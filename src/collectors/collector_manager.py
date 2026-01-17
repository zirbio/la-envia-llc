# src/collectors/collector_manager.py
"""Collector Manager for aggregating multiple social media collectors."""
import asyncio
from typing import AsyncIterator, Callable

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage


class CollectorManager:
    """Manages multiple collectors and aggregates their streams."""

    def __init__(self, collectors: list[BaseCollector]):
        """Initialize the collector manager.

        Args:
            collectors: List of collector instances to manage.
        """
        self._collectors = collectors
        self._callbacks: list[Callable[[SocialMessage], None]] = []
        self._running = False

    @property
    def collectors(self) -> list[BaseCollector]:
        """Return the list of managed collectors."""
        return self._collectors

    @property
    def callbacks(self) -> list[Callable[[SocialMessage], None]]:
        """Return the list of registered callbacks."""
        return self._callbacks

    def add_callback(self, callback: Callable[[SocialMessage], None]) -> None:
        """Add a callback to be called for each message.

        Args:
            callback: Function that takes a SocialMessage as argument.
        """
        self._callbacks.append(callback)

    async def connect_all(self) -> None:
        """Connect all collectors concurrently."""
        if self._collectors:
            await asyncio.gather(*[c.connect() for c in self._collectors])

    async def disconnect_all(self) -> None:
        """Disconnect all collectors."""
        self._running = False
        if self._collectors:
            await asyncio.gather(*[c.disconnect() for c in self._collectors])

    async def stream_all(self) -> AsyncIterator[SocialMessage]:
        """Stream messages from all collectors using asyncio.Queue.

        Yields:
            SocialMessage objects from all collectors as they arrive.
        """
        if not self._collectors:
            return

        self._running = True
        queue: asyncio.Queue[SocialMessage] = asyncio.Queue()

        async def collector_worker(collector: BaseCollector) -> None:
            """Worker that streams from a single collector into the queue."""
            try:
                async for msg in collector.stream():
                    if not self._running:
                        break
                    await queue.put(msg)
                    # Call all registered callbacks
                    for callback in self._callbacks:
                        try:
                            callback(msg)
                        except Exception:
                            # Don't let callback errors stop the stream
                            pass
            except Exception as e:
                # Log error but don't crash the worker
                print(f"Error in collector {collector.name}: {e}")

        # Create tasks for all collectors
        tasks = [asyncio.create_task(collector_worker(c)) for c in self._collectors]

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield msg
                except asyncio.TimeoutError:
                    # Check if all tasks are done (no more messages coming)
                    if all(t.done() for t in tasks):
                        break
        finally:
            self._running = False
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self) -> None:
        """Run the collector manager continuously.

        Connects all collectors, streams messages (processing via callbacks),
        then disconnects when complete.
        """
        await self.connect_all()
        try:
            async for msg in self.stream_all():
                pass  # Messages are processed via callbacks
        finally:
            await self.disconnect_all()
