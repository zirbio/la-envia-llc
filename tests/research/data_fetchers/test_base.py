# tests/research/data_fetchers/test_base.py

import pytest
from abc import ABC
from src.research.data_fetchers.base import BaseFetcher


class TestBaseFetcher:
    def test_is_abstract(self):
        assert issubclass(BaseFetcher, ABC)

        with pytest.raises(TypeError):
            BaseFetcher("test")

    def test_concrete_implementation(self):
        class ConcreteFetcher(BaseFetcher):
            async def fetch(self) -> dict:
                return {"data": "test"}

        fetcher = ConcreteFetcher("test_fetcher")
        assert fetcher.name == "test_fetcher"

    @pytest.mark.asyncio
    async def test_fetch_returns_dict(self):
        class ConcreteFetcher(BaseFetcher):
            async def fetch(self) -> dict:
                return {"key": "value"}

        fetcher = ConcreteFetcher("test")
        result = await fetcher.fetch()
        assert result == {"key": "value"}
