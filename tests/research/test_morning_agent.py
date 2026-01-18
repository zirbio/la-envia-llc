# tests/research/test_morning_agent.py

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock
from src.research.morning_agent import MorningResearchAgent
from src.research.models import DailyBrief


class TestMorningResearchAgent:
    def test_initialization(self):
        agent = MorningResearchAgent(
            grok_api_key="test_grok_key",
            claude_api_key="test_claude_key",
        )
        assert agent is not None

    @pytest.mark.asyncio
    async def test_generate_brief_returns_daily_brief(self):
        agent = MorningResearchAgent(
            grok_api_key="test_grok_key",
            claude_api_key="test_claude_key",
        )

        # Mock the fetchers
        mock_fetcher = MagicMock()
        mock_fetcher.name = "test_fetcher"
        mock_fetcher.fetch = AsyncMock(return_value={"test": "data"})
        agent._fetchers = [mock_fetcher]

        # Mock Claude response
        mock_response = {
            "market_regime": {"state": "risk-on", "trend": "bullish", "summary": "Test"},
            "ideas": [],
            "watchlist": [],
            "risks": [],
            "key_questions": [],
        }

        with patch.object(agent, "_call_claude", return_value=mock_response):
            brief = await agent.generate_brief(brief_type="initial")

        assert isinstance(brief, DailyBrief)
        assert brief.brief_type == "initial"

    @pytest.mark.asyncio
    async def test_fetchers_run_in_parallel(self):
        agent = MorningResearchAgent(
            grok_api_key="test_grok_key",
            claude_api_key="test_claude_key",
        )

        # Create mock fetchers
        mock_fetcher_1 = MagicMock()
        mock_fetcher_1.name = "fetcher1"
        mock_fetcher_1.fetch = AsyncMock(return_value={"data1": "value1"})

        mock_fetcher_2 = MagicMock()
        mock_fetcher_2.name = "fetcher2"
        mock_fetcher_2.fetch = AsyncMock(return_value={"data2": "value2"})

        agent._fetchers = [mock_fetcher_1, mock_fetcher_2]

        with patch.object(agent, "_call_claude", return_value={
            "market_regime": {"state": "neutral", "trend": "ranging", "summary": "Test"},
            "ideas": [], "watchlist": [], "risks": [], "key_questions": [],
        }):
            await agent.generate_brief(brief_type="initial")

        # Both fetchers should have been called
        mock_fetcher_1.fetch.assert_called_once()
        mock_fetcher_2.fetch.assert_called_once()
