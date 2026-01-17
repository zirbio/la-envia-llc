# src/collectors/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.models.social_message import SocialMessage, SourceType


class BaseCollector(ABC):
    """Abstract base class for all social media collectors."""

    def __init__(self, name: str, source_type: SourceType):
        self._name = name
        self._source_type = source_type
        self._connected = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass

    @abstractmethod
    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream messages from the source.

        Yields:
            SocialMessage objects as they arrive.
        """
        pass
