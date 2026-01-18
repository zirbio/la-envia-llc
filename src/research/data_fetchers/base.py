"""Base class for data fetchers."""

from abc import ABC, abstractmethod


class BaseFetcher(ABC):
    """Abstract base class for all data fetchers."""

    def __init__(self, name: str):
        """Initialize the fetcher.

        Args:
            name: Identifier for this fetcher.
        """
        self.name = name

    @abstractmethod
    async def fetch(self) -> dict:
        """Fetch data from the source.

        Returns:
            Dictionary containing fetched data.
        """
        pass
