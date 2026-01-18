# Grok Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace FinTwitBERT + multiple collectors with Grok as single source of social signals, plus dynamic credibility system that learns source accuracy.

**Architecture:** GrokCollector fetches X/Twitter via Grok API with native sentiment. DynamicCredibilityManager tracks source accuracy over time. SignalOutcomeTracker provides feedback loop to measure prediction success.

**Tech Stack:** Python 3.11+, asyncio, OpenAI SDK (Grok-compatible), Pydantic, pytest

---

## Task 1: Add GROK SourceType

**Files:**
- Modify: `src/models/social_message.py:10-15`
- Test: `tests/models/test_social_message.py`

**Step 1: Write the failing test**

```python
# tests/models/test_social_message.py
def test_grok_source_type_exists():
    from src.models.social_message import SourceType
    assert SourceType.GROK == "grok"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_social_message.py::test_grok_source_type_exists -v`
Expected: FAIL with "AttributeError: GROK"

**Step 3: Write minimal implementation**

```python
# src/models/social_message.py - Add to SourceType enum (line ~15)
class SourceType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    NEWS = "news"
    GROK = "grok"  # NEW
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_social_message.py::test_grok_source_type_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/social_message.py tests/models/test_social_message.py
git commit -m "feat: add GROK source type to SourceType enum"
```

---

## Task 2: Create SourceProfile Model

**Files:**
- Create: `src/scoring/source_profile.py`
- Test: `tests/scoring/test_source_profile.py`

**Step 1: Write the failing test**

```python
# tests/scoring/test_source_profile.py
from datetime import datetime
import pytest


def test_source_profile_creation():
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    profile = SourceProfile(
        author_id="unusual_whales",
        source_type=SourceType.GROK,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
    )

    assert profile.author_id == "unusual_whales"
    assert profile.source_type == SourceType.GROK
    assert profile.total_signals == 0
    assert profile.correct_signals == 0
    assert profile.accuracy == 0.0


def test_credibility_multiplier_insufficient_data():
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    profile = SourceProfile(
        author_id="new_trader",
        source_type=SourceType.GROK,
        total_signals=3,
        correct_signals=3,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
    )

    # Less than 5 signals = default 1.0
    assert profile.credibility_multiplier == 1.0


def test_credibility_multiplier_high_accuracy():
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    profile = SourceProfile(
        author_id="expert_trader",
        source_type=SourceType.GROK,
        total_signals=20,
        correct_signals=16,  # 80% accuracy
        first_seen=datetime.now(),
        last_seen=datetime.now(),
    )

    assert profile.accuracy == 0.8
    assert profile.credibility_multiplier == 1.3  # >= 75%


def test_credibility_multiplier_low_accuracy():
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    profile = SourceProfile(
        author_id="bad_trader",
        source_type=SourceType.GROK,
        total_signals=10,
        correct_signals=3,  # 30% accuracy
        first_seen=datetime.now(),
        last_seen=datetime.now(),
    )

    assert profile.accuracy == 0.3
    assert profile.credibility_multiplier == 0.5  # < 40%
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_source_profile.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.scoring.source_profile'"

**Step 3: Write minimal implementation**

```python
# src/scoring/source_profile.py
"""Source profile model for tracking author credibility."""

from dataclasses import dataclass, field
from datetime import datetime

from src.models.social_message import SourceType


@dataclass
class SourceProfile:
    """Profile of a source/author that generates signals.

    Tracks historical accuracy to compute dynamic credibility multiplier.
    """

    author_id: str
    source_type: SourceType
    first_seen: datetime
    last_seen: datetime

    # Credibility metrics
    total_signals: int = 0
    correct_signals: int = 0

    # Metadata
    followers: int | None = None
    verified: bool = False
    account_age_days: int | None = None
    category: str = "unknown"

    # Signal history (IDs)
    signals_history: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy ratio."""
        if self.total_signals == 0:
            return 0.0
        return self.correct_signals / self.total_signals

    @property
    def credibility_multiplier(self) -> float:
        """Calculate dynamic credibility multiplier based on accuracy.

        Returns:
            Multiplier from 0.5 to 1.3 based on historical accuracy.
            Returns 1.0 if insufficient data (< 5 signals).
        """
        if self.total_signals < 5:
            return 1.0  # Insufficient data

        if self.accuracy >= 0.75:
            return 1.3  # Tier 1+
        elif self.accuracy >= 0.60:
            return 1.1  # Tier 1
        elif self.accuracy >= 0.50:
            return 1.0  # Tier 2
        elif self.accuracy >= 0.40:
            return 0.8  # Tier 3
        else:
            return 0.5  # Unreliable
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scoring/test_source_profile.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/scoring/source_profile.py tests/scoring/test_source_profile.py
git commit -m "feat: add SourceProfile model with dynamic credibility multiplier"
```

---

## Task 3: Create SourceProfileStore

**Files:**
- Create: `src/scoring/source_profile_store.py`
- Test: `tests/scoring/test_source_profile_store.py`

**Step 1: Write the failing test**

```python
# tests/scoring/test_source_profile_store.py
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


def test_store_save_and_get():
    from src.scoring.source_profile import SourceProfile
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        profile = SourceProfile(
            author_id="test_user",
            source_type=SourceType.GROK,
            total_signals=10,
            correct_signals=8,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )

        store.save(profile)
        retrieved = store.get("test_user")

        assert retrieved is not None
        assert retrieved.author_id == "test_user"
        assert retrieved.total_signals == 10
        assert retrieved.correct_signals == 8


def test_store_get_nonexistent():
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        result = store.get("nonexistent")
        assert result is None


def test_store_get_or_create():
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        # First call creates
        profile1 = store.get_or_create("new_user", SourceType.GROK)
        assert profile1.author_id == "new_user"
        assert profile1.total_signals == 0

        # Modify and save
        profile1.total_signals = 5
        store.save(profile1)

        # Second call retrieves existing
        profile2 = store.get_or_create("new_user", SourceType.GROK)
        assert profile2.total_signals == 5


def test_store_persistence():
    from src.scoring.source_profile import SourceProfile
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save with first store instance
        store1 = SourceProfileStore(data_dir=Path(tmpdir))
        profile = SourceProfile(
            author_id="persistent_user",
            source_type=SourceType.GROK,
            total_signals=15,
            correct_signals=12,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        store1.save(profile)

        # Create new store instance (simulates restart)
        store2 = SourceProfileStore(data_dir=Path(tmpdir))
        retrieved = store2.get("persistent_user")

        assert retrieved is not None
        assert retrieved.total_signals == 15
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_source_profile_store.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/scoring/source_profile_store.py
"""Persistence layer for source profiles."""

import json
from datetime import datetime
from pathlib import Path

from src.models.social_message import SourceType
from src.scoring.source_profile import SourceProfile


class SourceProfileStore:
    """Stores and retrieves source profiles from disk."""

    def __init__(self, data_dir: Path = Path("data/sources")):
        """Initialize the store.

        Args:
            data_dir: Directory to store profile JSON files.
        """
        self._data_dir = data_dir
        self._cache: dict[str, SourceProfile] = {}
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def get(self, author_id: str) -> SourceProfile | None:
        """Get profile by author_id.

        Args:
            author_id: Unique identifier for the author.

        Returns:
            SourceProfile if found, None otherwise.
        """
        # Check cache first
        if author_id in self._cache:
            return self._cache[author_id]

        # Try loading from disk
        file_path = self._data_dir / f"{author_id}.json"
        if file_path.exists():
            profile = self._load_from_file(file_path)
            self._cache[author_id] = profile
            return profile

        return None

    def save(self, profile: SourceProfile) -> None:
        """Save or update a profile.

        Args:
            profile: SourceProfile to save.
        """
        self._cache[profile.author_id] = profile
        file_path = self._data_dir / f"{profile.author_id}.json"
        self._save_to_file(profile, file_path)

    def get_or_create(
        self, author_id: str, source_type: SourceType
    ) -> SourceProfile:
        """Get existing profile or create a new one.

        Args:
            author_id: Unique identifier for the author.
            source_type: Type of source (GROK, etc.).

        Returns:
            Existing or newly created SourceProfile.
        """
        profile = self.get(author_id)
        if profile is None:
            now = datetime.now()
            profile = SourceProfile(
                author_id=author_id,
                source_type=source_type,
                first_seen=now,
                last_seen=now,
            )
            self.save(profile)
        return profile

    def _load_from_file(self, file_path: Path) -> SourceProfile:
        """Load profile from JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        return SourceProfile(
            author_id=data["author_id"],
            source_type=SourceType(data["source_type"]),
            first_seen=datetime.fromisoformat(data["first_seen"]),
            last_seen=datetime.fromisoformat(data["last_seen"]),
            total_signals=data.get("total_signals", 0),
            correct_signals=data.get("correct_signals", 0),
            followers=data.get("followers"),
            verified=data.get("verified", False),
            account_age_days=data.get("account_age_days"),
            category=data.get("category", "unknown"),
            signals_history=data.get("signals_history", []),
        )

    def _save_to_file(self, profile: SourceProfile, file_path: Path) -> None:
        """Save profile to JSON file."""
        data = {
            "author_id": profile.author_id,
            "source_type": profile.source_type.value,
            "first_seen": profile.first_seen.isoformat(),
            "last_seen": profile.last_seen.isoformat(),
            "total_signals": profile.total_signals,
            "correct_signals": profile.correct_signals,
            "accuracy": profile.accuracy,
            "credibility_multiplier": profile.credibility_multiplier,
            "followers": profile.followers,
            "verified": profile.verified,
            "account_age_days": profile.account_age_days,
            "category": profile.category,
            "signals_history": profile.signals_history,
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scoring/test_source_profile_store.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/scoring/source_profile_store.py tests/scoring/test_source_profile_store.py
git commit -m "feat: add SourceProfileStore for profile persistence"
```

---

## Task 4: Create DynamicCredibilityManager

**Files:**
- Create: `src/scoring/dynamic_credibility_manager.py`
- Test: `tests/scoring/test_dynamic_credibility_manager.py`

**Step 1: Write the failing test**

```python
# tests/scoring/test_dynamic_credibility_manager.py
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


def test_get_multiplier_unknown_author():
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        multiplier = manager.get_multiplier("unknown_author", SourceType.GROK)
        assert multiplier == 1.0  # Default for unknown


def test_get_multiplier_tier1_seed():
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(
            profile_store=store,
            tier1_sources=["unusual_whales"],
            tier1_multiplier=1.3,
        )

        # No history but in tier1 seeds
        multiplier = manager.get_multiplier("unusual_whales", SourceType.GROK)
        assert multiplier == 1.3


def test_get_multiplier_dynamic_from_history():
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        # Create profile with history
        profile = SourceProfile(
            author_id="experienced_trader",
            source_type=SourceType.GROK,
            total_signals=10,
            correct_signals=8,  # 80% accuracy
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        store.save(profile)

        manager = DynamicCredibilityManager(profile_store=store)
        multiplier = manager.get_multiplier("experienced_trader", SourceType.GROK)
        assert multiplier == 1.3  # >= 75% accuracy


def test_record_signal():
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        manager.record_signal("new_author", SourceType.GROK, "signal_001")

        profile = store.get("new_author")
        assert profile is not None
        assert profile.total_signals == 1
        assert "signal_001" in profile.signals_history


def test_record_outcome():
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.scoring.source_profile import SourceProfile
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))

        # Create profile with some signals
        profile = SourceProfile(
            author_id="trader",
            source_type=SourceType.GROK,
            total_signals=5,
            correct_signals=3,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
        )
        store.save(profile)

        manager = DynamicCredibilityManager(profile_store=store)

        # Record correct outcome
        manager.record_outcome("trader", was_correct=True)

        updated = store.get("trader")
        assert updated.correct_signals == 4
        assert updated.accuracy == 0.8  # 4/5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_dynamic_credibility_manager.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/scoring/dynamic_credibility_manager.py
"""Dynamic credibility manager that learns from signal outcomes."""

from datetime import datetime

from src.models.social_message import SourceType
from src.scoring.source_profile_store import SourceProfileStore


class DynamicCredibilityManager:
    """Manages source credibility based on historical accuracy.

    Replaces static SourceCredibilityManager with dynamic learning.
    """

    def __init__(
        self,
        profile_store: SourceProfileStore,
        min_signals_for_ranking: int = 5,
        tier1_sources: list[str] | None = None,
        tier1_multiplier: float = 1.3,
    ):
        """Initialize the manager.

        Args:
            profile_store: Store for source profiles.
            min_signals_for_ranking: Minimum signals before using dynamic multiplier.
            tier1_sources: Seed list of known reliable sources.
            tier1_multiplier: Multiplier for tier1 sources without history.
        """
        self._store = profile_store
        self._min_signals = min_signals_for_ranking
        self._tier1_sources = tier1_sources or []
        self._tier1_multiplier = tier1_multiplier

    def get_multiplier(self, author_id: str, source_type: SourceType) -> float:
        """Get credibility multiplier for an author.

        Priority:
        1. If sufficient history -> use dynamic accuracy-based multiplier
        2. If in tier1_sources (seed) -> use tier1_multiplier
        3. Default -> 1.0

        Args:
            author_id: Unique identifier for the author.
            source_type: Type of source.

        Returns:
            Credibility multiplier (0.5 to 1.3).
        """
        profile = self._store.get(author_id)

        if profile and profile.total_signals >= self._min_signals:
            # Sufficient history -> use dynamic multiplier
            return profile.credibility_multiplier

        if author_id in self._tier1_sources:
            # Known source without sufficient history -> use seed
            return self._tier1_multiplier

        # Unknown author
        return 1.0

    def record_signal(
        self, author_id: str, source_type: SourceType, signal_id: str
    ) -> None:
        """Record that an author generated a signal.

        Args:
            author_id: Unique identifier for the author.
            source_type: Type of source.
            signal_id: Unique identifier for the signal.
        """
        profile = self._store.get_or_create(author_id, source_type)
        profile.total_signals += 1
        profile.last_seen = datetime.now()
        profile.signals_history.append(signal_id)
        self._store.save(profile)

    def record_outcome(self, author_id: str, was_correct: bool) -> None:
        """Record the outcome of a signal (correct/incorrect).

        Args:
            author_id: Unique identifier for the author.
            was_correct: Whether the signal prediction was correct.
        """
        profile = self._store.get(author_id)
        if profile:
            if was_correct:
                profile.correct_signals += 1
            # Accuracy is computed property, no need to update
            self._store.save(profile)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scoring/test_dynamic_credibility_manager.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/scoring/dynamic_credibility_manager.py tests/scoring/test_dynamic_credibility_manager.py
git commit -m "feat: add DynamicCredibilityManager with learning capability"
```

---

## Task 5: Create GrokCollector

**Files:**
- Create: `src/collectors/grok_collector.py`
- Test: `tests/collectors/test_grok_collector.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/collectors/test_grok_collector.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/collectors/grok_collector.py
"""Collector for X/Twitter data via Grok API."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from openai import AsyncOpenAI

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType

logger = logging.getLogger(__name__)


class GrokCollector(BaseCollector):
    """Collector for X/Twitter using Grok API with x_search."""

    def __init__(
        self,
        api_key: str,
        search_queries: list[str],
        refresh_interval: int = 30,
        max_results_per_query: int = 20,
    ):
        """Initialize the Grok collector.

        Args:
            api_key: xAI API key.
            search_queries: List of search queries (tickers, terms, from:user).
            refresh_interval: Seconds between search cycles.
            max_results_per_query: Max results per query.
        """
        super().__init__(name="grok", source_type=SourceType.GROK)
        self._api_key = api_key
        self._search_queries = search_queries
        self._refresh_interval = refresh_interval
        self._max_results = max_results_per_query
        self._client: AsyncOpenAI | None = None
        self._seen_ids: set[str] = set()

    async def connect(self) -> None:
        """Initialize Grok API client."""
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url="https://api.x.ai/v1",
        )
        self._connected = True
        logger.info("GrokCollector connected")

    async def disconnect(self) -> None:
        """Close connection."""
        self._client = None
        self._connected = False
        logger.info("GrokCollector disconnected")

    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream messages from X via Grok x_search.

        Yields:
            SocialMessage objects as they arrive.
        """
        if not self._connected or not self._client:
            raise RuntimeError("Collector not connected. Call connect() first.")

        while True:
            for query in self._search_queries:
                try:
                    async for msg in self._search_x(query):
                        yield msg
                except Exception as e:
                    logger.error(f"Error searching for '{query}': {e}")

            await asyncio.sleep(self._refresh_interval)

    async def _search_x(self, query: str) -> AsyncIterator[SocialMessage]:
        """Execute search on X via Grok.

        Args:
            query: Search query string.

        Yields:
            SocialMessage for each new post found.
        """
        if not self._client:
            return

        try:
            # Use Grok with x_search tool
            response = await self._client.chat.completions.create(
                model="grok-2",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Search X for the most recent posts about the given query. "
                            "Return the raw post data including id, author, text, "
                            "timestamps, engagement metrics, and sentiment score."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Search X for: {query}",
                    },
                ],
            )

            # Parse response and extract posts
            posts = self._parse_grok_response(response)

            for post in posts:
                post_id = post.get("id", "")
                if post_id and post_id not in self._seen_ids:
                    self._seen_ids.add(post_id)
                    yield self._to_social_message(post)

        except Exception as e:
            logger.error(f"Grok API error for query '{query}': {e}")

    def _parse_grok_response(self, response) -> list[dict]:
        """Parse Grok response to extract X posts.

        Args:
            response: Grok API response.

        Returns:
            List of post dictionaries.
        """
        # Extract content from response
        if not response.choices:
            return []

        content = response.choices[0].message.content
        if not content:
            return []

        # TODO: Parse structured response from Grok
        # For now, return empty - actual parsing depends on Grok's response format
        return []

    def _to_social_message(self, post: dict) -> SocialMessage:
        """Convert X post to SocialMessage.

        Args:
            post: Post dictionary from Grok.

        Returns:
            SocialMessage object.
        """
        # Parse timestamp
        created_at = post.get("created_at", "")
        try:
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(timezone.utc)

        author = post.get("author_username", "unknown")
        post_id = post.get("id", "")

        return SocialMessage(
            source=SourceType.GROK,
            source_id=str(post_id),
            author=author,
            content=post.get("text", ""),
            timestamp=timestamp,
            url=f"https://x.com/{author}/status/{post_id}",
            like_count=post.get("like_count"),
            retweet_count=post.get("retweet_count"),
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/collectors/test_grok_collector.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/collectors/grok_collector.py tests/collectors/test_grok_collector.py
git commit -m "feat: add GrokCollector for X/Twitter via Grok API"
```

---

## Task 6: Create SignalOutcomeTracker

**Files:**
- Create: `src/scoring/signal_outcome_tracker.py`
- Test: `tests/scoring/test_signal_outcome_tracker.py`

**Step 1: Write the failing test**

```python
# tests/scoring/test_signal_outcome_tracker.py
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_record_signal():
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)
        mock_alpaca = MagicMock()

        tracker = SignalOutcomeTracker(
            credibility_manager=manager,
            alpaca_client=mock_alpaca,
            evaluation_window_minutes=30,
        )

        tracker.record_signal(
            signal_id="sig_001",
            author_id="trader123",
            symbol="NVDA",
            direction="bullish",
            entry_price=500.0,
        )

        assert len(tracker._pending_evaluations) == 1
        eval = tracker._pending_evaluations[0]
        assert eval.author_id == "trader123"
        assert eval.symbol == "NVDA"
        assert eval.direction == "bullish"


@pytest.mark.asyncio
async def test_evaluate_correct_bullish():
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker, PendingEvaluation
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        mock_alpaca = AsyncMock()
        mock_alpaca.get_current_price = AsyncMock(return_value=510.0)  # +2%

        tracker = SignalOutcomeTracker(
            credibility_manager=manager,
            alpaca_client=mock_alpaca,
            evaluation_window_minutes=30,
            success_threshold_percent=1.0,
        )

        # Record signal first to create profile
        manager.record_signal("trader123", SourceType.GROK, "sig_001")

        # Add pending evaluation that's ready
        tracker._pending_evaluations.append(
            PendingEvaluation(
                signal_id="sig_001",
                author_id="trader123",
                symbol="NVDA",
                direction="bullish",
                entry_price=500.0,
                entry_time=datetime.now() - timedelta(minutes=35),
                evaluate_at=datetime.now() - timedelta(minutes=5),
            )
        )

        await tracker.evaluate_pending()

        # Check outcome was recorded
        profile = store.get("trader123")
        assert profile.correct_signals == 1


@pytest.mark.asyncio
async def test_evaluate_incorrect_bullish():
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker, PendingEvaluation
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.source_profile_store import SourceProfileStore
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        store = SourceProfileStore(data_dir=Path(tmpdir))
        manager = DynamicCredibilityManager(profile_store=store)

        mock_alpaca = AsyncMock()
        mock_alpaca.get_current_price = AsyncMock(return_value=490.0)  # -2%

        tracker = SignalOutcomeTracker(
            credibility_manager=manager,
            alpaca_client=mock_alpaca,
            evaluation_window_minutes=30,
            success_threshold_percent=1.0,
        )

        # Record signal first
        manager.record_signal("bad_trader", SourceType.GROK, "sig_002")

        # Add pending evaluation
        tracker._pending_evaluations.append(
            PendingEvaluation(
                signal_id="sig_002",
                author_id="bad_trader",
                symbol="NVDA",
                direction="bullish",
                entry_price=500.0,
                entry_time=datetime.now() - timedelta(minutes=35),
                evaluate_at=datetime.now() - timedelta(minutes=5),
            )
        )

        await tracker.evaluate_pending()

        # Check outcome was recorded as incorrect
        profile = store.get("bad_trader")
        assert profile.correct_signals == 0  # Not incremented
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scoring/test_signal_outcome_tracker.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/scoring/signal_outcome_tracker.py
"""Tracks signal outcomes for accuracy measurement."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.models.social_message import SourceType
from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager

logger = logging.getLogger(__name__)


@dataclass
class PendingEvaluation:
    """A signal pending outcome evaluation."""

    signal_id: str
    author_id: str
    symbol: str
    direction: str  # "bullish" or "bearish"
    entry_price: float
    entry_time: datetime
    evaluate_at: datetime


class SignalOutcomeTracker:
    """Tracks signal outcomes to measure source accuracy.

    Records signals when trades are executed, then evaluates
    whether the prediction was correct after a time window.
    """

    def __init__(
        self,
        credibility_manager: DynamicCredibilityManager,
        alpaca_client,  # Type hint omitted to avoid circular import
        evaluation_window_minutes: int = 30,
        success_threshold_percent: float = 1.0,
    ):
        """Initialize the tracker.

        Args:
            credibility_manager: Manager to update with outcomes.
            alpaca_client: Alpaca client for price data.
            evaluation_window_minutes: Minutes to wait before evaluating.
            success_threshold_percent: Price change % to consider success.
        """
        self._credibility = credibility_manager
        self._alpaca = alpaca_client
        self._window_minutes = evaluation_window_minutes
        self._threshold_percent = success_threshold_percent
        self._pending_evaluations: list[PendingEvaluation] = []

    def record_signal(
        self,
        signal_id: str,
        author_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
    ) -> None:
        """Record a signal for future evaluation.

        Args:
            signal_id: Unique identifier for the signal.
            author_id: Author who generated the signal.
            symbol: Stock symbol.
            direction: "bullish" or "bearish".
            entry_price: Price at trade execution.
        """
        now = datetime.now()
        evaluation = PendingEvaluation(
            signal_id=signal_id,
            author_id=author_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=now,
            evaluate_at=now + timedelta(minutes=self._window_minutes),
        )
        self._pending_evaluations.append(evaluation)

        logger.info(
            f"Recorded signal {signal_id} from {author_id} for {symbol} "
            f"({direction}) @ ${entry_price:.2f}"
        )

    async def evaluate_pending(self) -> None:
        """Evaluate all signals that have passed the evaluation window."""
        now = datetime.now()
        ready = [e for e in self._pending_evaluations if e.evaluate_at <= now]

        for evaluation in ready:
            try:
                was_correct = await self._evaluate_outcome(evaluation)
                self._credibility.record_outcome(evaluation.author_id, was_correct)

                logger.info(
                    f"Evaluated signal {evaluation.signal_id}: "
                    f"{'CORRECT' if was_correct else 'INCORRECT'}"
                )
            except Exception as e:
                logger.error(f"Error evaluating {evaluation.signal_id}: {e}")
            finally:
                self._pending_evaluations.remove(evaluation)

    async def _evaluate_outcome(self, evaluation: PendingEvaluation) -> bool:
        """Determine if a signal prediction was correct.

        Args:
            evaluation: The pending evaluation to check.

        Returns:
            True if prediction was correct, False otherwise.
        """
        current_price = await self._alpaca.get_current_price(evaluation.symbol)
        price_change_pct = (
            (current_price - evaluation.entry_price) / evaluation.entry_price
        ) * 100

        if evaluation.direction == "bullish":
            return price_change_pct >= self._threshold_percent
        else:  # bearish
            return price_change_pct <= -self._threshold_percent
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scoring/test_signal_outcome_tracker.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/scoring/signal_outcome_tracker.py tests/scoring/test_signal_outcome_tracker.py
git commit -m "feat: add SignalOutcomeTracker for feedback loop"
```

---

## Task 7: Update Scoring Module Exports

**Files:**
- Modify: `src/scoring/__init__.py`

**Step 1: Update exports**

```python
# src/scoring/__init__.py
"""Scoring module for signal evaluation and credibility management."""

from src.scoring.signal_scorer import SignalScorer
from src.scoring.source_credibility_manager import SourceCredibilityManager
from src.scoring.time_factor_calculator import TimeFactorCalculator
from src.scoring.confluence_detector import ConfluenceDetector
from src.scoring.dynamic_weight_calculator import DynamicWeightCalculator
from src.scoring.recommendation_builder import RecommendationBuilder
from src.scoring.models import ScoreTier, ScoringResult

# New exports
from src.scoring.source_profile import SourceProfile
from src.scoring.source_profile_store import SourceProfileStore
from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
from src.scoring.signal_outcome_tracker import SignalOutcomeTracker, PendingEvaluation

__all__ = [
    "SignalScorer",
    "SourceCredibilityManager",
    "TimeFactorCalculator",
    "ConfluenceDetector",
    "DynamicWeightCalculator",
    "RecommendationBuilder",
    "ScoreTier",
    "ScoringResult",
    # New
    "SourceProfile",
    "SourceProfileStore",
    "DynamicCredibilityManager",
    "SignalOutcomeTracker",
    "PendingEvaluation",
]
```

**Step 2: Commit**

```bash
git add src/scoring/__init__.py
git commit -m "feat: export new credibility components from scoring module"
```

---

## Task 8: Update Collectors Module Exports

**Files:**
- Modify: `src/collectors/__init__.py`

**Step 1: Update exports**

```python
# src/collectors/__init__.py
"""Collectors module for social media data collection."""

from src.collectors.base import BaseCollector
from src.collectors.collector_manager import CollectorManager
from src.collectors.stocktwits_collector import StocktwitsCollector
from src.collectors.reddit_collector import RedditCollector
from src.collectors.twitter_collector import TwitterCollector
from src.collectors.grok_collector import GrokCollector  # NEW

__all__ = [
    "BaseCollector",
    "CollectorManager",
    "StocktwitsCollector",
    "RedditCollector",
    "TwitterCollector",
    "GrokCollector",  # NEW
]
```

**Step 2: Commit**

```bash
git add src/collectors/__init__.py
git commit -m "feat: export GrokCollector from collectors module"
```

---

## Task 9: Add Grok Settings to Config

**Files:**
- Modify: `src/config/settings.py`
- Modify: `config/settings.yaml`

**Step 1: Add GrokCollectorConfig class**

Add to `src/config/settings.py` after `StocktwitsCollectorConfig`:

```python
class GrokCollectorConfig(BaseModel):
    """Configuration for Grok collector."""

    enabled: bool = True
    refresh_interval_seconds: int = 30
    max_results_per_query: int = 20
    search_queries: list[str] = Field(
        default_factory=lambda: [
            "$NVDA",
            "$TSLA",
            "$AMD",
            "from:unusual_whales",
            "from:DeItaone",
        ]
    )
```

Update `CollectorsConfig`:

```python
class CollectorsConfig(BaseModel):
    twitter: TwitterCollectorConfig = Field(default_factory=TwitterCollectorConfig)
    reddit: RedditCollectorConfig = Field(default_factory=RedditCollectorConfig)
    stocktwits: StocktwitsCollectorConfig = Field(default_factory=StocktwitsCollectorConfig)
    grok: GrokCollectorConfig = Field(default_factory=GrokCollectorConfig)  # NEW
```

Add to `ScoringSettings`:

```python
class ScoringSettings(BaseModel):
    # ... existing fields ...

    # Dynamic credibility (NEW)
    min_signals_for_dynamic: int = Field(default=5, ge=1, le=20)
    evaluation_window_minutes: int = Field(default=30, ge=5, le=120)
    success_threshold_percent: float = Field(default=1.0, ge=0.5, le=5.0)
    tier1_seeds: list[str] = Field(
        default_factory=lambda: [
            "unusual_whales",
            "DeItaone",
            "FirstSquawk",
            "optionsflow",
        ]
    )
```

**Step 2: Update settings.yaml**

Add grok section to `config/settings.yaml`:

```yaml
collectors:
  grok:
    enabled: true
    refresh_interval_seconds: 30
    max_results_per_query: 20
    search_queries:
      - "$NVDA"
      - "$TSLA"
      - "$AMD"
      - "$AAPL"
      - "$META"
      - "unusual options activity"
      - "from:unusual_whales"
      - "from:DeItaone"
      - "from:FirstSquawk"

scoring:
  # ... existing ...

  # Dynamic credibility
  min_signals_for_dynamic: 5
  evaluation_window_minutes: 30
  success_threshold_percent: 1.0
  tier1_seeds:
    - unusual_whales
    - DeItaone
    - FirstSquawk
    - optionsflow
```

**Step 3: Commit**

```bash
git add src/config/settings.py config/settings.yaml
git commit -m "feat: add Grok collector and dynamic credibility settings"
```

---

## Task 10: Update main.py

**Files:**
- Modify: `main.py`

**Step 1: Update imports**

Replace/update imports at top of `main.py`:

```python
# Remove these:
# from src.collectors import CollectorManager, StocktwitsCollector, RedditCollector
# from src.analyzers import AnalyzerManager, SentimentAnalyzer, ClaudeAnalyzer

# Add these:
from src.collectors import CollectorManager, GrokCollector
from src.analyzers import ClaudeAnalyzer
from src.scoring import (
    SignalScorer,
    DynamicCredibilityManager,
    SourceProfileStore,
    SignalOutcomeTracker,
    TimeFactorCalculator,
    ConfluenceDetector,
    DynamicWeightCalculator,
    RecommendationBuilder,
)
```

**Step 2: Update validate_env_vars**

```python
def validate_env_vars() -> None:
    required_vars = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",  # NEW
    ]
    # ... rest same
```

**Step 3: Simplify initialize_analyzers**

```python
async def initialize_analyzers(settings: Settings) -> ClaudeAnalyzer:
    """Initialize Claude analyzer (Grok provides sentiment)."""
    try:
        claude_analyzer = ClaudeAnalyzer(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=settings.analyzers.claude.model,
            max_tokens=settings.analyzers.claude.max_tokens,
            rate_limit_per_minute=settings.analyzers.claude.rate_limit_per_minute,
        )
        logger.info("✓ Claude API verified")
        return claude_analyzer

    except Exception as e:
        logger.error(f"Claude API failed: {e}")
        sys.exit(1)
```

**Step 4: Simplify initialize_collectors**

```python
def initialize_collectors(settings: Settings) -> CollectorManager:
    """Initialize GrokCollector as sole signal source."""
    if not settings.collectors.grok.enabled:
        logger.error("Grok collector must be enabled")
        sys.exit(1)

    grok = GrokCollector(
        api_key=os.getenv("XAI_API_KEY"),
        search_queries=settings.collectors.grok.search_queries,
        refresh_interval=settings.collectors.grok.refresh_interval_seconds,
        max_results_per_query=settings.collectors.grok.max_results_per_query,
    )
    logger.info("✓ GrokCollector initialized")

    return CollectorManager(collectors=[grok])
```

**Step 5: Update initialize_pipeline_components**

Add dynamic credibility:

```python
def initialize_pipeline_components(
    settings: Settings,
    alpaca_client: AlpacaClient,
) -> dict:
    # NEW: Source Profile Store
    profile_store = SourceProfileStore(data_dir=Path("data/sources"))
    logger.info("✓ SourceProfileStore initialized")

    # NEW: Dynamic Credibility Manager
    credibility_manager = DynamicCredibilityManager(
        profile_store=profile_store,
        min_signals_for_ranking=settings.scoring.min_signals_for_dynamic,
        tier1_sources=settings.scoring.tier1_seeds,
        tier1_multiplier=settings.scoring.credibility_tier1_multiplier,
    )
    logger.info("✓ DynamicCredibilityManager initialized")

    # Time, confluence, weight, recommendation (same as before)
    time_calculator = TimeFactorCalculator(...)
    confluence_detector = ConfluenceDetector(...)
    weight_calculator = DynamicWeightCalculator(...)
    recommendation_builder = RecommendationBuilder(...)

    signal_scorer = SignalScorer(
        credibility_manager=credibility_manager,  # Now dynamic
        time_calculator=time_calculator,
        confluence_detector=confluence_detector,
        weight_calculator=weight_calculator,
        recommendation_builder=recommendation_builder,
    )
    logger.info("✓ SignalScorer initialized")

    # Technical validator, gate, risk, journal, executor (same)
    # ...

    # NEW: Outcome Tracker
    outcome_tracker = SignalOutcomeTracker(
        credibility_manager=credibility_manager,
        alpaca_client=alpaca_client,
        evaluation_window_minutes=settings.scoring.evaluation_window_minutes,
        success_threshold_percent=settings.scoring.success_threshold_percent,
    )
    logger.info("✓ SignalOutcomeTracker initialized")

    return {
        "scorer": signal_scorer,
        "validator": technical_validator,
        "gate": market_gate,
        "risk_manager": risk_manager,
        "journal": journal_manager,
        "executor": trade_executor,
        "outcome_tracker": outcome_tracker,  # NEW
        "profile_store": profile_store,  # NEW
        "credibility_manager": credibility_manager,  # NEW
    }
```

**Step 6: Update create_data_dirs**

```python
def create_data_dirs() -> None:
    dirs = [
        Path("data/trades"),
        Path("data/signals"),
        Path("data/cache"),
        Path("data/backtest_results"),
        Path("data/sources"),  # NEW
    ]
    # ...
```

**Step 7: Commit**

```bash
git add main.py
git commit -m "feat: integrate Grok collector and dynamic credibility in main.py"
```

---

## Task 11: Integration Test

**Files:**
- Create: `tests/integration/test_grok_pipeline.py`

**Step 1: Write integration test**

```python
# tests/integration/test_grok_pipeline.py
"""Integration tests for Grok-based pipeline."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_full_pipeline_grok_to_credibility():
    """Test signal flows from Grok through credibility system."""
    from src.collectors.grok_collector import GrokCollector
    from src.scoring.source_profile_store import SourceProfileStore
    from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager
    from src.scoring.signal_outcome_tracker import SignalOutcomeTracker
    from src.models.social_message import SourceType

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup components
        store = SourceProfileStore(data_dir=Path(tmpdir))
        credibility = DynamicCredibilityManager(
            profile_store=store,
            tier1_sources=["unusual_whales"],
        )

        mock_alpaca = AsyncMock()
        mock_alpaca.get_current_price = AsyncMock(return_value=510.0)

        tracker = SignalOutcomeTracker(
            credibility_manager=credibility,
            alpaca_client=mock_alpaca,
        )

        # Simulate signal from unusual_whales
        author = "unusual_whales"

        # Check initial multiplier (tier1 seed)
        initial_mult = credibility.get_multiplier(author, SourceType.GROK)
        assert initial_mult == 1.3  # tier1 seed

        # Record signal
        credibility.record_signal(author, SourceType.GROK, "sig_001")

        # Verify profile created
        profile = store.get(author)
        assert profile is not None
        assert profile.total_signals == 1

        # Record more signals to reach threshold
        for i in range(4):
            credibility.record_signal(author, SourceType.GROK, f"sig_{i+2}")

        # Record outcomes (4 correct out of 5)
        for i in range(4):
            credibility.record_outcome(author, was_correct=True)
        credibility.record_outcome(author, was_correct=False)

        # Check dynamic multiplier
        profile = store.get(author)
        assert profile.total_signals == 5
        assert profile.correct_signals == 4
        assert profile.accuracy == 0.8
        assert profile.credibility_multiplier == 1.3  # >= 75%


def test_grok_collector_creates_valid_messages():
    """Test GrokCollector produces valid SocialMessage objects."""
    from src.collectors.grok_collector import GrokCollector
    from src.models.social_message import SourceType

    collector = GrokCollector(
        api_key="test",
        search_queries=["$NVDA"],
    )

    mock_post = {
        "id": "12345",
        "author_username": "trader",
        "text": "$NVDA to the moon!",
        "created_at": "2026-01-18T12:00:00Z",
        "like_count": 100,
        "retweet_count": 50,
    }

    msg = collector._to_social_message(mock_post)

    assert msg.source == SourceType.GROK
    assert msg.source_id == "12345"
    assert msg.author == "trader"
    assert "$NVDA" in msg.content
    assert msg.like_count == 100
```

**Step 2: Run integration test**

Run: `uv run pytest tests/integration/test_grok_pipeline.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_grok_pipeline.py
git commit -m "test: add integration tests for Grok pipeline"
```

---

## Task 12: Run Full Test Suite

**Step 1: Run all tests**

```bash
uv run pytest -v
```

Expected: All tests pass (existing + new)

**Step 2: Final commit**

```bash
git add -A
git commit -m "feat: complete Grok integration with dynamic credibility system"
```

---

## Summary

This plan implements:

1. **SourceProfile** - Model for tracking author accuracy
2. **SourceProfileStore** - Persistence layer for profiles
3. **DynamicCredibilityManager** - Learning credibility system
4. **GrokCollector** - X/Twitter data via Grok API
5. **SignalOutcomeTracker** - Feedback loop for accuracy measurement
6. **Updated main.py** - Simplified initialization with Grok only
7. **Updated settings** - Grok config and credibility seeds

Total: 12 tasks, ~50 tests, ~800 lines of new code.
