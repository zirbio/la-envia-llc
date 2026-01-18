# src/research/morning_agent.py

"""Morning Research Agent - orchestrates data fetching and analysis."""

import asyncio
import json
import logging
from datetime import datetime

from anthropic import Anthropic
from pydantic import ValidationError

from src.research.models import DailyBrief, MarketRegime, TradingIdea, WatchlistItem
from src.research.data_fetchers import MarketFetcher
from src.research.prompts import SYSTEM_PROMPT, build_context, TASK_PROMPT

logger = logging.getLogger(__name__)


class MorningResearchAgent:
    """Orchestrates morning research and generates Daily Brief."""

    def __init__(
        self,
        grok_api_key: str,
        claude_api_key: str,
        claude_model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4000,
    ):
        """Initialize the Morning Research Agent.

        Args:
            grok_api_key: API key for Grok (xAI).
            claude_api_key: API key for Claude (Anthropic).
            claude_model: Claude model to use for analysis.
            max_tokens: Maximum tokens for Claude response.
        """
        self._grok_api_key = grok_api_key
        self._claude_client = Anthropic(api_key=claude_api_key)
        self._claude_model = claude_model
        self._max_tokens = max_tokens

        # Initialize fetchers
        self._fetchers = [
            MarketFetcher(),
            # Add more fetchers here as they're implemented
        ]

    async def generate_brief(self, brief_type: str = "initial") -> DailyBrief:
        """Generate the Daily Brief.

        Args:
            brief_type: "initial" (6:00 AM ET) or "pre_open" (9:00 AM ET).

        Returns:
            Generated DailyBrief.

        Raises:
            ValueError: If brief_type is not "initial" or "pre_open".
        """
        # Validate brief_type parameter
        if brief_type not in ("initial", "pre_open"):
            raise ValueError(
                f"Invalid brief_type: {brief_type}. Must be 'initial' or 'pre_open'."
            )

        start_time = datetime.now()

        # Fetch all data in parallel
        fetch_start = datetime.now()
        fetched_data = await self._fetch_all_data()
        fetch_duration = (datetime.now() - fetch_start).total_seconds()

        # Add date/time to context
        now = datetime.now()
        fetched_data["date"] = now.strftime("%Y-%m-%d")
        fetched_data["time"] = now.strftime("%H:%M")

        # Call Claude for analysis
        analysis_start = datetime.now()
        analysis = await self._call_claude(fetched_data)
        analysis_duration = (datetime.now() - analysis_start).total_seconds()

        # Build DailyBrief
        brief = self._build_brief(
            analysis=analysis,
            brief_type=brief_type,
            data_sources=[f.name for f in self._fetchers],
            fetch_duration=fetch_duration,
            analysis_duration=analysis_duration,
        )

        logger.info(
            f"Generated {brief_type} brief in {(datetime.now() - start_time).total_seconds():.1f}s"
        )

        return brief

    async def _fetch_all_data(self) -> dict:
        """Fetch data from all sources in parallel.

        Returns:
            Combined dictionary of all fetched data.
        """
        tasks = [fetcher.fetch() for fetcher in self._fetchers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined = {}
        for fetcher, result in zip(self._fetchers, results):
            if isinstance(result, Exception):
                logger.error(f"Fetcher {fetcher.name} failed: {result}")
                combined[fetcher.name] = {"error": str(result)}
            else:
                combined.update(result)

        return combined

    async def _call_claude(self, data: dict) -> dict:
        """Call Claude to analyze data and generate brief.

        Args:
            data: Fetched data to analyze.

        Returns:
            Parsed JSON response from Claude.
        """
        context = build_context(data)

        # Run synchronous Claude API call in thread pool to avoid blocking event loop
        message = await asyncio.to_thread(
            self._claude_client.messages.create,
            model=self._claude_model,
            max_tokens=self._max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": f"{context}\n\n{TASK_PROMPT}",
                }
            ],
            system=SYSTEM_PROMPT,
        )

        # Extract and parse JSON from response
        content = message.content[0].text

        # Try to extract JSON from response
        try:
            # Find JSON in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response: {e}")

        # Return empty structure if parsing fails
        return {
            "market_regime": {"state": "neutral", "trend": "ranging", "summary": "Analysis unavailable"},
            "ideas": [],
            "watchlist": [],
            "risks": ["Unable to parse analysis"],
            "key_questions": [],
        }

    def _build_brief(
        self,
        analysis: dict,
        brief_type: str,
        data_sources: list[str],
        fetch_duration: float,
        analysis_duration: float,
    ) -> DailyBrief:
        """Build DailyBrief from Claude analysis.

        Args:
            analysis: Parsed analysis from Claude.
            brief_type: Type of brief.
            data_sources: List of data sources used.
            fetch_duration: Time spent fetching data.
            analysis_duration: Time spent on Claude analysis.

        Returns:
            Constructed DailyBrief.
        """
        # Parse market regime
        regime_data = analysis.get("market_regime", {})
        market_regime = MarketRegime(
            state=regime_data.get("state", "neutral"),
            trend=regime_data.get("trend", "ranging"),
            summary=regime_data.get("summary", ""),
        )

        # Parse ideas
        ideas = []
        for idea_data in analysis.get("ideas", []):
            try:
                idea = TradingIdea.model_validate(idea_data)
                ideas.append(idea)
            except ValidationError as e:
                logger.warning(f"Failed to parse idea: {e}")

        # Parse watchlist
        watchlist = []
        for item_data in analysis.get("watchlist", []):
            try:
                item = WatchlistItem.model_validate(item_data)
                watchlist.append(item)
            except ValidationError as e:
                logger.warning(f"Failed to parse watchlist item: {e}")

        return DailyBrief(
            generated_at=datetime.now(),
            brief_type=brief_type,
            market_regime=market_regime,
            ideas=ideas,
            watchlist=watchlist,
            risks=analysis.get("risks", []),
            key_questions=analysis.get("key_questions", []),
            data_sources_used=data_sources,
            fetch_duration_seconds=fetch_duration,
            analysis_duration_seconds=analysis_duration,
        )
