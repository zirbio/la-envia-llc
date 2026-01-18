# Morning Research Agent - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pre-market research agent that fetches data from multiple sources, uses Claude to synthesize a Daily Brief, and injects trading ideas into the existing system.

**Architecture:** Independent module (`src/research/`) with data fetchers running in parallel, Claude synthesis, and integration via `SocialMessage(source=RESEARCH)`. Scheduled at 12:00 and 15:00 Almería time.

**Tech Stack:** Python 3.13, Pydantic, asyncio, httpx, OpenAI client (for Grok), Anthropic client (for Claude), Streamlit (dashboard)

---

## Task 1: Add RESEARCH SourceType

**Files:**
- Modify: `src/models/social_message.py:10-15`
- Test: `tests/models/test_social_message.py`

**Step 1: Write the failing test**

```python
# tests/models/test_social_message.py - add to existing file

def test_source_type_research_exists():
    """Test that RESEARCH source type is available."""
    from src.models.social_message import SourceType
    assert SourceType.RESEARCH == "research"
    assert SourceType.RESEARCH.value == "research"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_social_message.py::test_source_type_research_exists -v`
Expected: FAIL with "AttributeError: RESEARCH"

**Step 3: Write minimal implementation**

```python
# src/models/social_message.py - modify SourceType enum

class SourceType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    GROK = "grok"
    RESEARCH = "research"  # Morning Research Agent
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_social_message.py::test_source_type_research_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/social_message.py tests/models/test_social_message.py
git commit -m "feat: add RESEARCH source type for Morning Research Agent"
```

---

## Task 2: Create TradingIdea Model

**Files:**
- Create: `src/research/__init__.py`
- Create: `src/research/models/__init__.py`
- Create: `src/research/models/trading_idea.py`
- Test: `tests/research/__init__.py`
- Test: `tests/research/models/__init__.py`
- Test: `tests/research/models/test_trading_idea.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/research/models
mkdir -p tests/research/models
touch src/research/__init__.py
touch src/research/models/__init__.py
touch tests/research/__init__.py
touch tests/research/models/__init__.py
```

**Step 2: Write the failing test**

```python
# tests/research/models/test_trading_idea.py

import pytest
from src.research.models.trading_idea import (
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
    TradingIdea,
)


class TestEnums:
    def test_direction_values(self):
        assert Direction.LONG == "LONG"
        assert Direction.SHORT == "SHORT"

    def test_conviction_values(self):
        assert Conviction.HIGH == "HIGH"
        assert Conviction.MEDIUM == "MEDIUM"
        assert Conviction.LOW == "LOW"

    def test_position_size_values(self):
        assert PositionSize.FULL == "FULL"
        assert PositionSize.HALF == "HALF"
        assert PositionSize.QUARTER == "QUARTER"


class TestTechnicalLevels:
    def test_create_technical_levels(self):
        levels = TechnicalLevels(
            support=135.50,
            resistance=142.00,
            entry_zone=(136.00, 137.50),
        )
        assert levels.support == 135.50
        assert levels.resistance == 142.00
        assert levels.entry_zone == (136.00, 137.50)


class TestRiskReward:
    def test_create_risk_reward(self):
        rr = RiskReward(
            entry=137.00,
            stop=134.50,
            target=145.00,
            ratio="3.2:1",
        )
        assert rr.entry == 137.00
        assert rr.stop == 134.50
        assert rr.target == 145.00
        assert rr.ratio == "3.2:1"


class TestTradingIdea:
    def test_create_trading_idea(self):
        idea = TradingIdea(
            rank=1,
            ticker="NVDA",
            direction=Direction.LONG,
            conviction=Conviction.HIGH,
            catalyst="TSMC beat + AI guidance raise",
            thesis="Semiconductor demand exceeding expectations",
            technical=TechnicalLevels(
                support=135.50,
                resistance=142.00,
                entry_zone=(136.00, 137.50),
            ),
            risk_reward=RiskReward(
                entry=137.00,
                stop=134.50,
                target=145.00,
                ratio="3.2:1",
            ),
            position_size=PositionSize.FULL,
            kill_switch="China export restrictions headline",
        )
        assert idea.ticker == "NVDA"
        assert idea.direction == Direction.LONG
        assert idea.conviction == Conviction.HIGH

    def test_trading_idea_json_serialization(self):
        idea = TradingIdea(
            rank=1,
            ticker="NVDA",
            direction=Direction.LONG,
            conviction=Conviction.HIGH,
            catalyst="Test catalyst",
            thesis="Test thesis",
            technical=TechnicalLevels(
                support=135.50,
                resistance=142.00,
                entry_zone=(136.00, 137.50),
            ),
            risk_reward=RiskReward(
                entry=137.00,
                stop=134.50,
                target=145.00,
                ratio="3.2:1",
            ),
            position_size=PositionSize.FULL,
            kill_switch="Test kill switch",
        )
        json_data = idea.model_dump_json()
        assert "NVDA" in json_data
        assert "LONG" in json_data
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/research/models/test_trading_idea.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# src/research/models/trading_idea.py

"""Trading idea models for Morning Research Agent."""

from enum import Enum
from pydantic import BaseModel


class Direction(str, Enum):
    """Trade direction."""
    LONG = "LONG"
    SHORT = "SHORT"


class Conviction(str, Enum):
    """Conviction level for trading idea."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class PositionSize(str, Enum):
    """Recommended position size."""
    FULL = "FULL"        # 1.0x normal size
    HALF = "HALF"        # 0.5x
    QUARTER = "QUARTER"  # 0.25x


class TechnicalLevels(BaseModel):
    """Key technical levels for the trade."""
    support: float
    resistance: float
    entry_zone: tuple[float, float]


class RiskReward(BaseModel):
    """Risk/reward parameters for the trade."""
    entry: float
    stop: float
    target: float
    ratio: str  # e.g., "2.5:1"


class TradingIdea(BaseModel):
    """A single trading idea from the Daily Brief."""
    rank: int
    ticker: str
    direction: Direction
    conviction: Conviction
    catalyst: str
    thesis: str
    technical: TechnicalLevels
    risk_reward: RiskReward
    position_size: PositionSize
    kill_switch: str
```

```python
# src/research/models/__init__.py

"""Research models."""

from src.research.models.trading_idea import (
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
    TradingIdea,
)

__all__ = [
    "Direction",
    "Conviction",
    "PositionSize",
    "TechnicalLevels",
    "RiskReward",
    "TradingIdea",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/research/models/test_trading_idea.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/research/ tests/research/
git commit -m "feat: add TradingIdea model for research agent"
```

---

## Task 3: Create DailyBrief Model

**Files:**
- Create: `src/research/models/daily_brief.py`
- Modify: `src/research/models/__init__.py`
- Test: `tests/research/models/test_daily_brief.py`

**Step 1: Write the failing test**

```python
# tests/research/models/test_daily_brief.py

import pytest
from datetime import datetime
from src.research.models.daily_brief import (
    MarketRegime,
    WatchlistItem,
    DailyBrief,
)
from src.research.models.trading_idea import (
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
    TradingIdea,
)


class TestMarketRegime:
    def test_create_market_regime(self):
        regime = MarketRegime(
            state="risk-on",
            trend="bullish",
            summary="Markets trending higher on strong earnings",
        )
        assert regime.state == "risk-on"
        assert regime.trend == "bullish"


class TestWatchlistItem:
    def test_create_watchlist_item(self):
        item = WatchlistItem(
            ticker="MSFT",
            setup="Consolidating near highs",
            trigger="Break above $420 with volume",
        )
        assert item.ticker == "MSFT"


class TestDailyBrief:
    def test_create_daily_brief(self):
        idea = TradingIdea(
            rank=1,
            ticker="NVDA",
            direction=Direction.LONG,
            conviction=Conviction.HIGH,
            catalyst="Test",
            thesis="Test",
            technical=TechnicalLevels(
                support=135.0, resistance=145.0, entry_zone=(136.0, 138.0)
            ),
            risk_reward=RiskReward(
                entry=137.0, stop=134.0, target=145.0, ratio="2.7:1"
            ),
            position_size=PositionSize.FULL,
            kill_switch="Test",
        )

        brief = DailyBrief(
            generated_at=datetime(2026, 1, 18, 12, 0, 0),
            brief_type="initial",
            market_regime=MarketRegime(
                state="risk-on",
                trend="bullish",
                summary="Test summary",
            ),
            ideas=[idea],
            watchlist=[
                WatchlistItem(ticker="MSFT", setup="Test", trigger="Test")
            ],
            risks=["CPI release at 14:30"],
            key_questions=["Will NVDA hold $135 support?"],
            data_sources_used=["grok", "sec", "yahoo"],
            fetch_duration_seconds=5.2,
            analysis_duration_seconds=12.3,
        )

        assert brief.brief_type == "initial"
        assert len(brief.ideas) == 1
        assert brief.ideas[0].ticker == "NVDA"

    def test_daily_brief_json_serialization(self):
        brief = DailyBrief(
            generated_at=datetime(2026, 1, 18, 12, 0, 0),
            brief_type="pre_open",
            market_regime=MarketRegime(
                state="neutral",
                trend="ranging",
                summary="Test",
            ),
            ideas=[],
            watchlist=[],
            risks=[],
            key_questions=[],
            data_sources_used=[],
            fetch_duration_seconds=0.0,
            analysis_duration_seconds=0.0,
        )
        json_data = brief.model_dump_json()
        assert "pre_open" in json_data
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/research/models/test_daily_brief.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/research/models/daily_brief.py

"""Daily Brief model for Morning Research Agent."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel

from src.research.models.trading_idea import TradingIdea


class MarketRegime(BaseModel):
    """Current market regime assessment."""
    state: Literal["risk-on", "risk-off", "neutral"]
    trend: Literal["bullish", "bearish", "ranging"]
    summary: str


class WatchlistItem(BaseModel):
    """A ticker to watch but not trade today."""
    ticker: str
    setup: str
    trigger: str


class DailyBrief(BaseModel):
    """The complete Daily Brief from Morning Research Agent."""
    generated_at: datetime
    brief_type: Literal["initial", "pre_open"]
    timezone: str = "Europe/Madrid"

    market_regime: MarketRegime
    ideas: list[TradingIdea]
    watchlist: list[WatchlistItem]
    risks: list[str]
    key_questions: list[str]

    # Metadata
    data_sources_used: list[str]
    fetch_duration_seconds: float
    analysis_duration_seconds: float
```

```python
# src/research/models/__init__.py - update

"""Research models."""

from src.research.models.trading_idea import (
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
    TradingIdea,
)
from src.research.models.daily_brief import (
    MarketRegime,
    WatchlistItem,
    DailyBrief,
)

__all__ = [
    "Direction",
    "Conviction",
    "PositionSize",
    "TechnicalLevels",
    "RiskReward",
    "TradingIdea",
    "MarketRegime",
    "WatchlistItem",
    "DailyBrief",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/research/models/test_daily_brief.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/research/models/ tests/research/models/
git commit -m "feat: add DailyBrief model for research agent"
```

---

## Task 4: Create Integration Module (TradingIdea → SocialMessage)

**Files:**
- Create: `src/research/integration.py`
- Test: `tests/research/test_integration.py`

**Step 1: Write the failing test**

```python
# tests/research/test_integration.py

import pytest
from datetime import datetime
from src.research.integration import idea_to_social_message
from src.research.models import (
    TradingIdea,
    DailyBrief,
    MarketRegime,
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
)
from src.models.social_message import SourceType


class TestIdeaToSocialMessage:
    @pytest.fixture
    def sample_idea(self):
        return TradingIdea(
            rank=1,
            ticker="NVDA",
            direction=Direction.LONG,
            conviction=Conviction.HIGH,
            catalyst="TSMC beat + AI guidance",
            thesis="Strong demand",
            technical=TechnicalLevels(
                support=135.0, resistance=145.0, entry_zone=(136.0, 138.0)
            ),
            risk_reward=RiskReward(
                entry=137.0, stop=134.0, target=145.0, ratio="2.7:1"
            ),
            position_size=PositionSize.FULL,
            kill_switch="China export ban",
        )

    @pytest.fixture
    def sample_brief(self, sample_idea):
        return DailyBrief(
            generated_at=datetime(2026, 1, 18, 12, 0, 0),
            brief_type="initial",
            market_regime=MarketRegime(
                state="risk-on", trend="bullish", summary="Test"
            ),
            ideas=[sample_idea],
            watchlist=[],
            risks=[],
            key_questions=[],
            data_sources_used=["grok"],
            fetch_duration_seconds=1.0,
            analysis_duration_seconds=2.0,
        )

    def test_converts_to_social_message(self, sample_idea, sample_brief):
        msg = idea_to_social_message(sample_idea, sample_brief)

        assert msg.source == SourceType.RESEARCH
        assert "NVDA" in msg.content
        assert "LONG" in msg.content
        assert msg.author == "morning_research_agent"

    def test_source_id_is_unique(self, sample_idea, sample_brief):
        msg = idea_to_social_message(sample_idea, sample_brief)

        assert "brief_" in msg.source_id
        assert "NVDA" in msg.source_id

    def test_metadata_includes_trade_params(self, sample_idea, sample_brief):
        msg = idea_to_social_message(sample_idea, sample_brief)

        assert msg.metadata["conviction"] == "HIGH"
        assert msg.metadata["direction"] == "LONG"
        assert msg.metadata["entry"] == 137.0
        assert msg.metadata["stop"] == 134.0
        assert msg.metadata["target"] == 145.0

    def test_content_includes_risk_reward(self, sample_idea, sample_brief):
        msg = idea_to_social_message(sample_idea, sample_brief)

        assert "Entry: $137.0" in msg.content
        assert "Stop: $134.0" in msg.content
        assert "Target: $145.0" in msg.content
        assert "2.7:1" in msg.content
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/research/test_integration.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/research/integration.py

"""Integration with existing trading system."""

from src.models.social_message import SocialMessage, SourceType
from src.research.models import TradingIdea, DailyBrief


def idea_to_social_message(idea: TradingIdea, brief: DailyBrief) -> SocialMessage:
    """Convert a TradingIdea to SocialMessage for the trading system.

    Args:
        idea: The trading idea to convert.
        brief: The parent DailyBrief for context.

    Returns:
        SocialMessage that can be processed by the trading system.
    """
    content = (
        f"${idea.ticker} {idea.direction.value} - {idea.conviction.value} conviction. "
        f"Catalyst: {idea.catalyst}. "
        f"Entry: ${idea.risk_reward.entry}, Stop: ${idea.risk_reward.stop}, "
        f"Target: ${idea.risk_reward.target} (R:R {idea.risk_reward.ratio}). "
        f"Kill switch: {idea.kill_switch}"
    )

    return SocialMessage(
        source=SourceType.RESEARCH,
        source_id=f"brief_{brief.generated_at.isoformat()}_{idea.ticker}",
        author="morning_research_agent",
        content=content,
        timestamp=brief.generated_at,
        url=None,
        metadata={
            "rank": idea.rank,
            "conviction": idea.conviction.value,
            "direction": idea.direction.value,
            "entry": idea.risk_reward.entry,
            "stop": idea.risk_reward.stop,
            "target": idea.risk_reward.target,
            "position_size": idea.position_size.value,
        },
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/research/test_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/research/integration.py tests/research/test_integration.py
git commit -m "feat: add integration module to convert ideas to SocialMessage"
```

---

## Task 5: Create Base Fetcher Abstract Class

**Files:**
- Create: `src/research/data_fetchers/__init__.py`
- Create: `src/research/data_fetchers/base.py`
- Test: `tests/research/data_fetchers/__init__.py`
- Test: `tests/research/data_fetchers/test_base.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/research/data_fetchers
mkdir -p tests/research/data_fetchers
touch src/research/data_fetchers/__init__.py
touch tests/research/data_fetchers/__init__.py
```

**Step 2: Write the failing test**

```python
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
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/research/data_fetchers/test_base.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# src/research/data_fetchers/base.py

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
```

```python
# src/research/data_fetchers/__init__.py

"""Data fetchers for Morning Research Agent."""

from src.research.data_fetchers.base import BaseFetcher

__all__ = ["BaseFetcher"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/research/data_fetchers/test_base.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/research/data_fetchers/ tests/research/data_fetchers/
git commit -m "feat: add BaseFetcher abstract class"
```

---

## Task 6: Create Market Fetcher (Futures, VIX, Gappers)

**Files:**
- Create: `src/research/data_fetchers/market_fetcher.py`
- Modify: `src/research/data_fetchers/__init__.py`
- Test: `tests/research/data_fetchers/test_market_fetcher.py`

**Step 1: Write the failing test**

```python
# tests/research/data_fetchers/test_market_fetcher.py

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.research.data_fetchers.market_fetcher import MarketFetcher


class TestMarketFetcher:
    def test_inherits_base_fetcher(self):
        from src.research.data_fetchers.base import BaseFetcher
        fetcher = MarketFetcher()
        assert isinstance(fetcher, BaseFetcher)
        assert fetcher.name == "market"

    @pytest.mark.asyncio
    async def test_fetch_returns_expected_structure(self):
        fetcher = MarketFetcher()

        # Mock yfinance
        with patch("src.research.data_fetchers.market_fetcher.yf") as mock_yf:
            # Mock futures data
            mock_es = MagicMock()
            mock_es.info = {"regularMarketPrice": 5000.0, "regularMarketChangePercent": 0.5}

            mock_nq = MagicMock()
            mock_nq.info = {"regularMarketPrice": 17500.0, "regularMarketChangePercent": 0.8}

            mock_vix = MagicMock()
            mock_vix.info = {"regularMarketPrice": 14.5}

            mock_yf.Ticker.side_effect = lambda x: {
                "ES=F": mock_es,
                "NQ=F": mock_nq,
                "^VIX": mock_vix,
            }.get(x, MagicMock())

            result = await fetcher.fetch()

        assert "futures" in result
        assert "vix" in result
        assert "gappers" in result

    @pytest.mark.asyncio
    async def test_fetch_handles_errors_gracefully(self):
        fetcher = MarketFetcher()

        with patch("src.research.data_fetchers.market_fetcher.yf") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("API Error")

            result = await fetcher.fetch()

        # Should return empty/default structure, not raise
        assert "futures" in result
        assert "error" in result or result["futures"]["es"] is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/research/data_fetchers/test_market_fetcher.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/research/data_fetchers/market_fetcher.py

"""Fetcher for market data (futures, VIX, pre-market gappers)."""

import logging
import yfinance as yf

from src.research.data_fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)


class MarketFetcher(BaseFetcher):
    """Fetches futures, VIX, and pre-market gapper data."""

    def __init__(
        self,
        min_gap_percent: float = 3.0,
        min_volume: int = 100000,
    ):
        """Initialize the market fetcher.

        Args:
            min_gap_percent: Minimum gap percentage for gappers.
            min_volume: Minimum pre-market volume for gappers.
        """
        super().__init__("market")
        self.min_gap_percent = min_gap_percent
        self.min_volume = min_volume

    async def fetch(self) -> dict:
        """Fetch market data.

        Returns:
            Dictionary with futures, VIX, and gapper data.
        """
        result = {
            "futures": {"es": None, "nq": None, "es_change": None, "nq_change": None},
            "vix": None,
            "gappers": [],
        }

        try:
            # Fetch futures
            es = yf.Ticker("ES=F")
            nq = yf.Ticker("NQ=F")
            vix = yf.Ticker("^VIX")

            es_info = es.info
            nq_info = nq.info
            vix_info = vix.info

            result["futures"] = {
                "es": es_info.get("regularMarketPrice"),
                "nq": nq_info.get("regularMarketPrice"),
                "es_change": es_info.get("regularMarketChangePercent"),
                "nq_change": nq_info.get("regularMarketChangePercent"),
            }

            result["vix"] = vix_info.get("regularMarketPrice")

            # Fetch gappers (simplified - would use screener in production)
            result["gappers"] = await self._fetch_gappers()

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            result["error"] = str(e)

        return result

    async def _fetch_gappers(self) -> list[dict]:
        """Fetch pre-market gappers.

        Returns:
            List of gapper dictionaries.
        """
        # In production, this would scrape Finviz or use a screener API
        # For now, return empty list - will be enhanced later
        return []
```

**Step 4: Update __init__.py**

```python
# src/research/data_fetchers/__init__.py

"""Data fetchers for Morning Research Agent."""

from src.research.data_fetchers.base import BaseFetcher
from src.research.data_fetchers.market_fetcher import MarketFetcher

__all__ = ["BaseFetcher", "MarketFetcher"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/research/data_fetchers/test_market_fetcher.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/research/data_fetchers/ tests/research/data_fetchers/
git commit -m "feat: add MarketFetcher for futures, VIX, and gappers"
```

---

## Task 7: Create Prompts Module

**Files:**
- Create: `src/research/prompts/__init__.py`
- Create: `src/research/prompts/templates.py`
- Test: `tests/research/prompts/__init__.py`
- Test: `tests/research/prompts/test_templates.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/research/prompts
mkdir -p tests/research/prompts
touch src/research/prompts/__init__.py
touch tests/research/prompts/__init__.py
```

**Step 2: Write the failing test**

```python
# tests/research/prompts/test_templates.py

import pytest
from src.research.prompts.templates import (
    SYSTEM_PROMPT,
    build_context,
    TASK_PROMPT,
)


class TestSystemPrompt:
    def test_system_prompt_exists(self):
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_principles(self):
        assert "PRINCIPLES" in SYSTEM_PROMPT
        assert "evidence" in SYSTEM_PROMPT.lower()

    def test_system_prompt_contains_rejection_criteria(self):
        assert "REJECTION" in SYSTEM_PROMPT


class TestBuildContext:
    def test_build_context_with_data(self):
        data = {
            "date": "2026-01-18",
            "time": "06:00",
            "futures": {"es": 5000.0, "nq": 17500.0, "es_change": 0.5, "nq_change": 0.8},
            "vix": 14.5,
            "gappers": [{"ticker": "NVDA", "gap_percent": 5.0, "volume": 500000}],
            "earnings": ["NFLX", "TSLA"],
            "economic_events": ["CPI at 14:30"],
            "sec_filings": {"8k": [], "form4": []},
            "social_intelligence": "Unusual activity in NVDA options",
            "news": ["Tech earnings strong"],
        }

        context = build_context(data)

        assert "2026-01-18" in context
        assert "5000.0" in context
        assert "NVDA" in context
        assert "CPI" in context

    def test_build_context_handles_missing_data(self):
        data = {"date": "2026-01-18", "time": "06:00"}

        context = build_context(data)

        assert "2026-01-18" in context
        # Should not raise, should use defaults


class TestTaskPrompt:
    def test_task_prompt_exists(self):
        assert TASK_PROMPT is not None
        assert len(TASK_PROMPT) > 100

    def test_task_prompt_contains_sections(self):
        assert "MARKET REGIME" in TASK_PROMPT
        assert "TOP IDEAS" in TASK_PROMPT
        assert "WATCHLIST" in TASK_PROMPT
        assert "RISKS" in TASK_PROMPT

    def test_task_prompt_contains_json_schema(self):
        assert "output_format" in TASK_PROMPT.lower() or "json" in TASK_PROMPT.lower()
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/research/prompts/test_templates.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# src/research/prompts/templates.py

"""Prompt templates for Morning Research Agent."""

SYSTEM_PROMPT = """You are a senior equity research analyst at a quantitative hedge fund.
Your job: produce the Morning Brief before US market open.

PRINCIPLES:
1. Every conclusion must cite evidence (data source + specific numbers)
2. Distinguish FACT (data) from INFERENCE (your analysis)
3. Think contrarian: what is the market missing or mispricing?
4. Express conviction in probabilities, not certainties
5. Focus on asymmetric risk/reward (2:1 minimum)
6. If data is insufficient, say "INSUFFICIENT DATA" - never guess

REJECTION CRITERIA (automatic NO):
- No clear catalyst in next 24-48 hours
- Risk/reward below 1.5:1
- Conflicting signals without resolution
- Low liquidity (< 500K avg volume)"""


def build_context(data: dict) -> str:
    """Build context string from fetched data.

    Args:
        data: Dictionary with fetched market data.

    Returns:
        Formatted context string for the prompt.
    """
    futures = data.get("futures", {})
    gappers = data.get("gappers", [])
    earnings = data.get("earnings", [])
    economic = data.get("economic_events", [])
    sec = data.get("sec_filings", {"8k": [], "form4": []})
    social = data.get("social_intelligence", "No social data available")
    news = data.get("news", [])

    # Format gappers table
    gappers_table = ""
    for g in gappers:
        gappers_table += f"| {g.get('ticker', 'N/A')} | {g.get('gap_percent', 0):.1f}% | {g.get('volume', 0):,} | {g.get('catalyst', 'Unknown')} |\n"
    if not gappers_table:
        gappers_table = "| No significant gappers found |\n"

    return f"""<context>
<market_snapshot>
Date: {data.get('date', 'Unknown')} | Time: {data.get('time', 'Unknown')} ET
ES Futures: {futures.get('es', 'N/A')} ({futures.get('es_change', 0):.1f}%)
NQ Futures: {futures.get('nq', 'N/A')} ({futures.get('nq_change', 0):.1f}%)
VIX: {data.get('vix', 'N/A')}
</market_snapshot>

<premarket_gappers>
| Ticker | Gap% | Volume | Catalyst |
|--------|------|--------|----------|
{gappers_table}</premarket_gappers>

<earnings_today>
{', '.join(earnings) if earnings else 'No major earnings today'}
</earnings_today>

<economic_calendar>
{chr(10).join(economic) if economic else 'No major events scheduled'}
</economic_calendar>

<sec_filings_24h>
8-K Material Events: {len(sec.get('8k', []))} filings
Form 4 Insider Activity: {len(sec.get('form4', []))} filings
</sec_filings_24h>

<social_intelligence>
{social}
</social_intelligence>

<overnight_news>
{chr(10).join(news) if news else 'No significant overnight news'}
</overnight_news>
</context>"""


TASK_PROMPT = """Analyze ALL data above. Produce today's MORNING BRIEF.

REQUIRED SECTIONS:

1. MARKET REGIME (3-4 sentences)
   - Current market state (risk-on/off, trending/ranging)
   - Key overnight developments
   - What the VIX and futures are signaling

2. TOP IDEAS (max 5, ranked by conviction)
   For EACH idea provide:
   - Ticker, Direction (LONG/SHORT), Conviction (HIGH/MED/LOW)
   - Catalyst: What's driving this TODAY? (specific event/data)
   - Thesis: Why will price move in your direction?
   - Technical: Key levels (support, resistance, entry zone)
   - Risk/Reward: Entry, Stop, Target (with R:R ratio)
   - Position Size: FULL (1x) / HALF (0.5x) / QUARTER (0.25x)
   - Kill Switch: What would invalidate this idea?

3. WATCHLIST (3-5 tickers)
   - Setting up but not actionable today
   - What trigger would make them tradeable?

4. RISKS & LANDMINES
   - Events that could blow up positions
   - Scheduled times to be careful

5. KEY QUESTIONS
   - What information would change your thesis today?

Return ONLY valid JSON matching this schema:
{
  "market_regime": {
    "state": "risk-on|risk-off|neutral",
    "trend": "bullish|bearish|ranging",
    "summary": "string"
  },
  "ideas": [
    {
      "rank": 1,
      "ticker": "NVDA",
      "direction": "LONG",
      "conviction": "HIGH",
      "catalyst": "string",
      "thesis": "string",
      "technical": {
        "support": 135.50,
        "resistance": 142.00,
        "entry_zone": [136.00, 137.50]
      },
      "risk_reward": {
        "entry": 137.00,
        "stop": 134.50,
        "target": 145.00,
        "ratio": "3.2:1"
      },
      "position_size": "FULL",
      "kill_switch": "string"
    }
  ],
  "watchlist": [
    {"ticker": "MSFT", "setup": "string", "trigger": "string"}
  ],
  "risks": ["string"],
  "key_questions": ["string"]
}"""
```

```python
# src/research/prompts/__init__.py

"""Prompts for Morning Research Agent."""

from src.research.prompts.templates import (
    SYSTEM_PROMPT,
    build_context,
    TASK_PROMPT,
)

__all__ = ["SYSTEM_PROMPT", "build_context", "TASK_PROMPT"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/research/prompts/test_templates.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/research/prompts/ tests/research/prompts/
git commit -m "feat: add prompt templates for Claude analysis"
```

---

## Task 8: Create Morning Agent Orchestrator

**Files:**
- Create: `src/research/morning_agent.py`
- Modify: `src/research/__init__.py`
- Test: `tests/research/test_morning_agent.py`

**Step 1: Write the failing test**

```python
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
        agent._fetchers = [AsyncMock(fetch=AsyncMock(return_value={"test": "data"}))]

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/research/test_morning_agent.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/research/morning_agent.py

"""Morning Research Agent - orchestrates data fetching and analysis."""

import asyncio
import json
import logging
from datetime import datetime

from anthropic import Anthropic

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
        """
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

        message = self._claude_client.messages.create(
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
            except Exception as e:
                logger.warning(f"Failed to parse idea: {e}")

        # Parse watchlist
        watchlist = []
        for item_data in analysis.get("watchlist", []):
            try:
                item = WatchlistItem.model_validate(item_data)
                watchlist.append(item)
            except Exception as e:
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
```

```python
# src/research/__init__.py

"""Morning Research Agent module."""

from src.research.models import (
    Direction,
    Conviction,
    PositionSize,
    TechnicalLevels,
    RiskReward,
    TradingIdea,
    MarketRegime,
    WatchlistItem,
    DailyBrief,
)
from src.research.integration import idea_to_social_message
from src.research.morning_agent import MorningResearchAgent

__all__ = [
    "Direction",
    "Conviction",
    "PositionSize",
    "TechnicalLevels",
    "RiskReward",
    "TradingIdea",
    "MarketRegime",
    "WatchlistItem",
    "DailyBrief",
    "idea_to_social_message",
    "MorningResearchAgent",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/research/test_morning_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/research/ tests/research/
git commit -m "feat: add MorningResearchAgent orchestrator"
```

---

## Task 9: Add Research Configuration to Settings

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`
- Test: `tests/config/test_settings.py` (add new test)

**Step 1: Write the failing test**

```python
# tests/config/test_research_settings.py

import pytest
from src.config.settings import Settings, ResearchConfig


class TestResearchConfig:
    def test_research_config_exists(self):
        assert ResearchConfig is not None

    def test_research_config_fields(self):
        config = ResearchConfig(
            enabled=True,
            timezone="Europe/Madrid",
            initial_brief_time="12:00",
            pre_open_brief_time="15:00",
            claude_model="claude-sonnet-4-20250514",
            max_tokens=4000,
            max_ideas=5,
            max_watchlist=5,
            briefs_dir="data/research/briefs",
            inject_to_orchestrator=True,
            telegram_enabled=True,
        )

        assert config.enabled is True
        assert config.timezone == "Europe/Madrid"
        assert config.initial_brief_time == "12:00"

    def test_settings_has_research(self):
        # This will fail until we add research to Settings
        settings = Settings.from_yaml("config/settings.yaml")
        assert hasattr(settings, "research")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_research_settings.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `src/config/settings.py`:

```python
# Add this class to src/config/settings.py

@dataclass
class ResearchConfig:
    """Configuration for Morning Research Agent."""
    enabled: bool = True
    timezone: str = "Europe/Madrid"
    initial_brief_time: str = "12:00"
    pre_open_brief_time: str = "15:00"
    claude_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4000
    max_ideas: int = 5
    max_watchlist: int = 5
    briefs_dir: str = "data/research/briefs"
    inject_to_orchestrator: bool = True
    telegram_enabled: bool = True
    telegram_summary: bool = True


# Add to Settings class:
# research: ResearchConfig
```

Add to `config/settings.yaml`:

```yaml
# Morning Research Agent
research:
  enabled: true
  timezone: "Europe/Madrid"
  initial_brief_time: "12:00"
  pre_open_brief_time: "15:00"
  claude_model: "claude-sonnet-4-20250514"
  max_tokens: 4000
  max_ideas: 5
  max_watchlist: 5
  briefs_dir: "data/research/briefs"
  inject_to_orchestrator: true
  telegram_enabled: true
  telegram_summary: true
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_research_settings.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/config/settings.py config/settings.yaml tests/config/
git commit -m "feat: add research configuration to settings"
```

---

## Task 10: Create Dashboard Page

**Files:**
- Create: `src/dashboard/pages/5_Research.py`
- Test: Manual testing via `uv run streamlit run src/dashboard/Home.py`

**Step 1: Write the dashboard page**

```python
# src/dashboard/pages/5_Research.py

"""Research Dashboard - Morning Brief visualization."""

import json
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Research", page_icon="📊", layout="wide")

st.title("📊 Morning Research Brief")


def load_briefs(briefs_dir: Path, date: datetime) -> dict:
    """Load briefs for a specific date."""
    date_str = date.strftime("%Y-%m-%d")
    briefs = {}

    initial_path = briefs_dir / f"{date_str}_initial.json"
    pre_open_path = briefs_dir / f"{date_str}_pre_open.json"

    if initial_path.exists():
        with open(initial_path) as f:
            briefs["initial"] = json.load(f)

    if pre_open_path.exists():
        with open(pre_open_path) as f:
            briefs["pre_open"] = json.load(f)

    return briefs


# Date selector
col1, col2 = st.columns([2, 1])
with col1:
    selected_date = st.date_input("Select Date", value=datetime.now().date())
with col2:
    brief_type = st.selectbox("Brief Type", ["pre_open", "initial"])

# Load briefs
briefs_dir = Path("data/research/briefs")
briefs = load_briefs(briefs_dir, datetime.combine(selected_date, datetime.min.time()))

if brief_type in briefs:
    brief = briefs[brief_type]

    # Market Regime
    regime = brief.get("market_regime", {})
    state_emoji = {"risk-on": "🟢", "risk-off": "🔴", "neutral": "🟡"}.get(regime.get("state", ""), "⚪")

    st.markdown(f"""
    ### Market Regime {state_emoji}
    **State:** {regime.get('state', 'N/A')} | **Trend:** {regime.get('trend', 'N/A')}

    {regime.get('summary', 'No summary available')}
    """)

    st.divider()

    # Top Ideas
    st.subheader("🎯 Top Ideas")

    ideas = brief.get("ideas", [])
    if ideas:
        for idea in ideas:
            conviction_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "gray"}.get(
                idea.get("conviction", ""), "gray"
            )
            direction_emoji = "⬆️" if idea.get("direction") == "LONG" else "⬇️"

            with st.expander(
                f"#{idea.get('rank', '?')} {idea.get('ticker', 'N/A')} {idea.get('direction', '')} {direction_emoji} "
                f"[{idea.get('conviction', 'N/A')}] - {idea.get('position_size', 'N/A')} Position"
            ):
                st.markdown(f"""
                **Catalyst:** {idea.get('catalyst', 'N/A')}

                **Thesis:** {idea.get('thesis', 'N/A')}

                **Technical Levels:**
                - Support: ${idea.get('technical', {}).get('support', 'N/A')}
                - Resistance: ${idea.get('technical', {}).get('resistance', 'N/A')}
                - Entry Zone: {idea.get('technical', {}).get('entry_zone', 'N/A')}

                **Risk/Reward:**
                - Entry: ${idea.get('risk_reward', {}).get('entry', 'N/A')}
                - Stop: ${idea.get('risk_reward', {}).get('stop', 'N/A')}
                - Target: ${idea.get('risk_reward', {}).get('target', 'N/A')}
                - R:R: {idea.get('risk_reward', {}).get('ratio', 'N/A')}

                **Kill Switch:** {idea.get('kill_switch', 'N/A')}
                """)
    else:
        st.info("No trading ideas for this date")

    st.divider()

    # Watchlist and Risks
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👀 Watchlist")
        watchlist = brief.get("watchlist", [])
        if watchlist:
            for item in watchlist:
                st.markdown(f"**{item.get('ticker', 'N/A')}** - {item.get('setup', '')}")
                st.caption(f"Trigger: {item.get('trigger', 'N/A')}")
        else:
            st.info("No watchlist items")

    with col2:
        st.subheader("⚠️ Risks")
        risks = brief.get("risks", [])
        if risks:
            for risk in risks:
                st.markdown(f"• {risk}")
        else:
            st.info("No risks identified")

    st.divider()

    # Key Questions
    st.subheader("❓ Key Questions")
    questions = brief.get("key_questions", [])
    if questions:
        for q in questions:
            st.markdown(f"• {q}")
    else:
        st.info("No key questions")

    # Metadata
    st.divider()
    st.caption(
        f"Generated: {brief.get('generated_at', 'N/A')} | "
        f"Fetch: {brief.get('fetch_duration_seconds', 0):.1f}s | "
        f"Analysis: {brief.get('analysis_duration_seconds', 0):.1f}s | "
        f"Sources: {', '.join(brief.get('data_sources_used', []))}"
    )

else:
    st.warning(f"No {brief_type} brief found for {selected_date}")
    st.info("Briefs are generated at 12:00 (initial) and 15:00 (pre_open) Almería time.")
```

**Step 2: Test manually**

Run: `uv run streamlit run src/dashboard/Home.py --server.port 8501`

Navigate to the Research page and verify layout.

**Step 3: Commit**

```bash
git add src/dashboard/pages/5_Research.py
git commit -m "feat: add Research dashboard page for Daily Brief visualization"
```

---

## Task 11: Run Full Test Suite and Final Commit

**Step 1: Run all tests**

```bash
uv run pytest --tb=short -q
```

Expected: All tests pass (711+ tests)

**Step 2: Final commit and summary**

```bash
git add -A
git status
```

If there are any uncommitted changes:

```bash
git commit -m "chore: final cleanup for Morning Research Agent"
```

**Step 3: View implementation summary**

```bash
git log --oneline feature/morning-research-agent ^main
```

---

## Summary of Implementation

| Task | Component | Files |
|------|-----------|-------|
| 1 | RESEARCH SourceType | `src/models/social_message.py` |
| 2 | TradingIdea Model | `src/research/models/trading_idea.py` |
| 3 | DailyBrief Model | `src/research/models/daily_brief.py` |
| 4 | Integration Module | `src/research/integration.py` |
| 5 | Base Fetcher | `src/research/data_fetchers/base.py` |
| 6 | Market Fetcher | `src/research/data_fetchers/market_fetcher.py` |
| 7 | Prompts Module | `src/research/prompts/templates.py` |
| 8 | Morning Agent | `src/research/morning_agent.py` |
| 9 | Configuration | `src/config/settings.py`, `config/settings.yaml` |
| 10 | Dashboard | `src/dashboard/pages/5_Research.py` |

**Additional fetchers to implement later:**
- `grok_fetcher.py` - X/Twitter via Grok
- `sec_fetcher.py` - SEC EDGAR filings
- `earnings_fetcher.py` - Earnings calendar
- `economic_fetcher.py` - Economic events
- `news_fetcher.py` - Alpaca news

---

*Plan created: 2026-01-18*
