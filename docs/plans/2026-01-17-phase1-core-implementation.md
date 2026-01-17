# Phase 1: Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational data collection layer with Twitter, Reddit, Stocktwits collectors and Alpaca API connection.

**Architecture:** Event-driven collectors that stream social media data to a unified message queue. Each collector runs independently, extracting tickers and basic metadata. Alpaca client provides market data and news streaming.

**Tech Stack:** Python 3.11+, asyncio, twscrape, asyncpraw, pytwits, alpaca-py, pydantic, pytest-asyncio

---

## Prerequisites

### Task 0: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `config/settings.yaml`
- Create: `.env.example`

**Step 1: Create requirements.txt**

```txt
# Core
alpaca-py>=0.21.0
asyncpraw>=7.7.0
twscrape>=0.12.0
pytwits>=0.0.5
pydantic>=2.5.0
pydantic-settings>=2.1.0
pyyaml>=6.0.0
python-dotenv>=1.0.0

# Async
aiohttp>=3.9.0
aiofiles>=23.2.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0

# Dev
black>=23.12.0
ruff>=0.1.0
mypy>=1.8.0
```

**Step 2: Create pyproject.toml**

```toml
[project]
name = "intraday-trading-system"
version = "0.1.0"
description = "AI-powered intraday trading system with social media analysis"
requires-python = ">=3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]

[tool.black]
line-length = 88
target-version = ["py311"]

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.11"
strict = true
```

**Step 3: Create directory structure**

```bash
mkdir -p src/collectors src/models src/config tests/collectors tests/models config
touch src/__init__.py src/collectors/__init__.py src/models/__init__.py src/config/__init__.py
touch tests/__init__.py tests/collectors/__init__.py tests/models/__init__.py
```

**Step 4: Create .env.example**

```env
# Alpaca API
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_PAPER=true

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=TradingBot/1.0

# Telegram (for later)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Step 5: Create virtual environment and install**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 6: Commit**

```bash
git add -A
git commit -m "chore: initial project setup with dependencies"
```

---

## Task 1: Core Models - SocialMessage

**Files:**
- Create: `src/models/social_message.py`
- Test: `tests/models/test_social_message.py`

**Step 1: Write the failing test**

```python
# tests/models/test_social_message.py
import pytest
from datetime import datetime, timezone
from src.models.social_message import SocialMessage, SourceType


class TestSocialMessage:
    def test_create_twitter_message(self):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="tweet_123",
            author="unusual_whales",
            content="🚨 Large $NVDA call sweep",
            timestamp=datetime.now(timezone.utc),
            url="https://twitter.com/unusual_whales/status/123",
        )
        assert msg.source == SourceType.TWITTER
        assert msg.author == "unusual_whales"
        assert "$NVDA" in msg.content

    def test_create_reddit_message(self):
        msg = SocialMessage(
            source=SourceType.REDDIT,
            source_id="post_abc",
            author="trader123",
            content="DD on $AAPL earnings play",
            timestamp=datetime.now(timezone.utc),
            url="https://reddit.com/r/stocks/abc",
            subreddit="stocks",
        )
        assert msg.source == SourceType.REDDIT
        assert msg.subreddit == "stocks"

    def test_extract_tickers_from_content(self):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="test",
            content="Looking at $AAPL $MSFT and $GOOGL today",
            timestamp=datetime.now(timezone.utc),
        )
        tickers = msg.extract_tickers()
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_extract_tickers_no_duplicates(self):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="test",
            content="$NVDA calls, $NVDA puts, all $NVDA",
            timestamp=datetime.now(timezone.utc),
        )
        tickers = msg.extract_tickers()
        assert tickers == ["NVDA"]

    def test_extract_tickers_excludes_crypto(self):
        msg = SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="test",
            content="$BTC $ETH $AAPL $DOGE",
            timestamp=datetime.now(timezone.utc),
        )
        tickers = msg.extract_tickers(exclude_crypto=True)
        assert tickers == ["AAPL"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_social_message.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.models.social_message'"

**Step 3: Write minimal implementation**

```python
# src/models/social_message.py
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    NEWS = "news"


# Common crypto tickers to exclude
CRYPTO_TICKERS = {
    "BTC", "ETH", "DOGE", "SOL", "XRP", "ADA", "AVAX", "DOT",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "BCH", "SHIB", "PEPE"
}


class SocialMessage(BaseModel):
    """Represents a message from any social media source."""

    source: SourceType
    source_id: str
    author: str
    content: str
    timestamp: datetime
    url: Optional[str] = None

    # Reddit-specific
    subreddit: Optional[str] = None
    upvotes: Optional[int] = None
    comment_count: Optional[int] = None

    # Twitter-specific
    retweet_count: Optional[int] = None
    like_count: Optional[int] = None

    # Stocktwits-specific
    sentiment: Optional[str] = None  # "bullish" or "bearish"

    # Extracted data (populated later)
    extracted_tickers: list[str] = Field(default_factory=list)

    def extract_tickers(self, exclude_crypto: bool = True) -> list[str]:
        """Extract stock tickers from content.

        Args:
            exclude_crypto: If True, excludes common crypto tickers.

        Returns:
            List of unique tickers in order of appearance.
        """
        # Match $TICKER pattern (1-5 uppercase letters)
        pattern = r'\$([A-Z]{1,5})\b'
        matches = re.findall(pattern, self.content)

        # Remove duplicates while preserving order
        seen = set()
        tickers = []
        for ticker in matches:
            if ticker not in seen:
                if exclude_crypto and ticker in CRYPTO_TICKERS:
                    continue
                seen.add(ticker)
                tickers.append(ticker)

        return tickers
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_social_message.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/models/social_message.py tests/models/test_social_message.py
git commit -m "feat(models): add SocialMessage with ticker extraction"
```

---

## Task 2: Base Collector Interface

**Files:**
- Create: `src/collectors/base.py`
- Test: `tests/collectors/test_base.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/collectors/test_base.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/collectors/test_base.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/collectors/base.py tests/collectors/test_base.py
git commit -m "feat(collectors): add BaseCollector abstract interface"
```

---

## Task 3: Configuration System

**Files:**
- Create: `src/config/settings.py`
- Create: `config/settings.yaml`
- Test: `tests/config/test_settings.py`

**Step 1: Write the failing test**

```python
# tests/config/test_settings.py
import pytest
import tempfile
import os
from pathlib import Path


class TestSettings:
    def test_load_settings_from_yaml(self, tmp_path):
        # Create temp config file
        config_content = """
system:
  name: "Test System"
  mode: "paper"

collectors:
  twitter:
    enabled: true
    refresh_interval_seconds: 15
  reddit:
    enabled: true
    use_streaming: true

risk:
  circuit_breakers:
    per_trade:
      max_loss_percent: 1.0
    daily:
      max_loss_percent: 3.0
"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content)

        from src.config.settings import Settings
        settings = Settings.from_yaml(config_file)

        assert settings.system.name == "Test System"
        assert settings.system.mode == "paper"
        assert settings.collectors.twitter.enabled is True
        assert settings.collectors.twitter.refresh_interval_seconds == 15
        assert settings.risk.circuit_breakers.per_trade.max_loss_percent == 1.0

    def test_settings_defaults(self, tmp_path):
        # Minimal config
        config_content = """
system:
  name: "Minimal"
"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content)

        from src.config.settings import Settings
        settings = Settings.from_yaml(config_file)

        # Check defaults are applied
        assert settings.system.mode == "paper"  # default
        assert settings.collectors.twitter.enabled is True  # default

    def test_env_override(self, tmp_path, monkeypatch):
        config_content = """
system:
  name: "Test"
"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content)

        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")

        from src.config.settings import Settings
        settings = Settings.from_yaml(config_file)

        assert settings.alpaca.api_key == "test_key"
        assert settings.alpaca.secret_key == "test_secret"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_settings.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create config directory init**

```bash
mkdir -p src/config tests/config
touch src/config/__init__.py tests/config/__init__.py
```

**Step 4: Write minimal implementation**

```python
# src/config/settings.py
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class SystemConfig(BaseModel):
    name: str = "Intraday Trading System"
    version: str = "1.0.0"
    mode: str = "paper"  # paper | live
    timezone: str = "America/New_York"


class TwitterCollectorConfig(BaseModel):
    enabled: bool = True
    engine: str = "twscrape"
    accounts_pool_size: int = 5
    rate_limit_buffer: float = 0.8
    refresh_interval_seconds: int = 15


class RedditCollectorConfig(BaseModel):
    enabled: bool = True
    use_streaming: bool = True
    batch_fallback_interval: int = 60


class StocktwitsCollectorConfig(BaseModel):
    enabled: bool = True
    refresh_interval_seconds: int = 30


class CollectorsConfig(BaseModel):
    twitter: TwitterCollectorConfig = Field(default_factory=TwitterCollectorConfig)
    reddit: RedditCollectorConfig = Field(default_factory=RedditCollectorConfig)
    stocktwits: StocktwitsCollectorConfig = Field(default_factory=StocktwitsCollectorConfig)


class PerTradeRiskConfig(BaseModel):
    max_loss_percent: float = 1.0
    hard_stop: bool = True


class DailyRiskConfig(BaseModel):
    max_loss_percent: float = 3.0
    max_trades_after_loss: int = 0
    cooldown_minutes: int = 60


class WeeklyRiskConfig(BaseModel):
    max_loss_percent: float = 6.0
    force_paper_mode: bool = True


class CircuitBreakersConfig(BaseModel):
    per_trade: PerTradeRiskConfig = Field(default_factory=PerTradeRiskConfig)
    daily: DailyRiskConfig = Field(default_factory=DailyRiskConfig)
    weekly: WeeklyRiskConfig = Field(default_factory=WeeklyRiskConfig)


class RiskConfig(BaseModel):
    circuit_breakers: CircuitBreakersConfig = Field(default_factory=CircuitBreakersConfig)


class AlpacaConfig(BaseSettings):
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    paper_url: str = "https://paper-api.alpaca.markets"
    live_url: str = "https://api.alpaca.markets"

    class Config:
        env_prefix = "ALPACA_"


class RedditAPIConfig(BaseSettings):
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "TradingBot/1.0"

    class Config:
        env_prefix = "REDDIT_"


class Settings(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    alpaca: AlpacaConfig = Field(default_factory=AlpacaConfig)
    reddit_api: RedditAPIConfig = Field(default_factory=RedditAPIConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file with env var overrides."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Load env-based configs
        alpaca = AlpacaConfig()
        reddit_api = RedditAPIConfig()

        # Merge YAML with defaults
        return cls(
            **data,
            alpaca=alpaca,
            reddit_api=reddit_api,
        )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/config/test_settings.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add src/config/settings.py tests/config/test_settings.py
git commit -m "feat(config): add Settings with YAML loading and env overrides"
```

---

## Task 4: Twitter Collector

**Files:**
- Create: `src/collectors/twitter_collector.py`
- Test: `tests/collectors/test_twitter_collector.py`

**Step 1: Write the failing test**

```python
# tests/collectors/test_twitter_collector.py
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from src.collectors.twitter_collector import TwitterCollector
from src.models.social_message import SourceType


class TestTwitterCollector:
    @pytest.fixture
    def collector(self):
        accounts = ["unusual_whales", "FirstSquawk"]
        return TwitterCollector(accounts_to_follow=accounts)

    def test_collector_has_correct_source_type(self, collector):
        assert collector.source_type == SourceType.TWITTER

    def test_collector_has_correct_name(self, collector):
        assert collector.name == "twitter"

    def test_collector_stores_accounts(self, collector):
        assert "unusual_whales" in collector.accounts_to_follow
        assert "FirstSquawk" in collector.accounts_to_follow

    @pytest.mark.asyncio
    async def test_parse_tweet_to_message(self, collector):
        mock_tweet = MagicMock()
        mock_tweet.id = 123456789
        mock_tweet.user.username = "unusual_whales"
        mock_tweet.rawContent = "🚨 Large $NVDA call sweep $2.4M"
        mock_tweet.date = datetime.now(timezone.utc)
        mock_tweet.url = "https://twitter.com/unusual_whales/status/123456789"
        mock_tweet.retweetCount = 100
        mock_tweet.likeCount = 500

        msg = collector._parse_tweet(mock_tweet)

        assert msg.source == SourceType.TWITTER
        assert msg.source_id == "123456789"
        assert msg.author == "unusual_whales"
        assert "$NVDA" in msg.content
        assert msg.retweet_count == 100
        assert msg.like_count == 500

    @pytest.mark.asyncio
    async def test_extract_tickers_from_parsed_message(self, collector):
        mock_tweet = MagicMock()
        mock_tweet.id = 123
        mock_tweet.user.username = "test"
        mock_tweet.rawContent = "Looking at $AAPL $MSFT today"
        mock_tweet.date = datetime.now(timezone.utc)
        mock_tweet.url = "https://twitter.com/test/123"
        mock_tweet.retweetCount = 0
        mock_tweet.likeCount = 0

        msg = collector._parse_tweet(mock_tweet)
        tickers = msg.extract_tickers()

        assert "AAPL" in tickers
        assert "MSFT" in tickers
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/collectors/test_twitter_collector.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/collectors/twitter_collector.py
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType


class TwitterCollector(BaseCollector):
    """Collector for Twitter/X using twscrape."""

    def __init__(
        self,
        accounts_to_follow: list[str],
        refresh_interval: int = 15,
    ):
        super().__init__(name="twitter", source_type=SourceType.TWITTER)
        self._accounts_to_follow = accounts_to_follow
        self._refresh_interval = refresh_interval
        self._api = None
        self._last_tweet_ids: dict[str, int] = {}

    @property
    def accounts_to_follow(self) -> list[str]:
        return self._accounts_to_follow

    async def connect(self) -> None:
        """Initialize twscrape API connection."""
        try:
            from twscrape import API
            self._api = API()
            self._connected = True
        except ImportError:
            raise RuntimeError("twscrape not installed. Run: pip install twscrape")

    async def disconnect(self) -> None:
        """Close connection."""
        self._api = None
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream tweets from followed accounts.

        Note: twscrape doesn't support true streaming, so this
        polls for new tweets at the configured interval.
        """
        if not self._connected or self._api is None:
            raise RuntimeError("Collector not connected. Call connect() first.")

        for username in self._accounts_to_follow:
            async for tweet in self._get_user_tweets(username):
                yield self._parse_tweet(tweet)

    async def _get_user_tweets(self, username: str, limit: int = 10):
        """Get recent tweets from a user."""
        if self._api is None:
            return

        try:
            user = await self._api.user_by_login(username)
            if user:
                async for tweet in self._api.user_tweets(user.id, limit=limit):
                    # Skip if we've already seen this tweet
                    last_id = self._last_tweet_ids.get(username, 0)
                    if tweet.id > last_id:
                        self._last_tweet_ids[username] = tweet.id
                        yield tweet
        except Exception:
            # Log error but continue with other accounts
            pass

    def _parse_tweet(self, tweet) -> SocialMessage:
        """Convert a twscrape Tweet to SocialMessage."""
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id=str(tweet.id),
            author=tweet.user.username,
            content=tweet.rawContent,
            timestamp=tweet.date if tweet.date.tzinfo else tweet.date.replace(tzinfo=timezone.utc),
            url=tweet.url,
            retweet_count=tweet.retweetCount,
            like_count=tweet.likeCount,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/collectors/test_twitter_collector.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/collectors/twitter_collector.py tests/collectors/test_twitter_collector.py
git commit -m "feat(collectors): add TwitterCollector with twscrape"
```

---

## Task 5: Reddit Collector

**Files:**
- Create: `src/collectors/reddit_collector.py`
- Test: `tests/collectors/test_reddit_collector.py`

**Step 1: Write the failing test**

```python
# tests/collectors/test_reddit_collector.py
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from src.collectors.reddit_collector import RedditCollector
from src.models.social_message import SourceType


class TestRedditCollector:
    @pytest.fixture
    def collector(self):
        subreddits = ["wallstreetbets", "stocks"]
        return RedditCollector(
            subreddits=subreddits,
            client_id="test_id",
            client_secret="test_secret",
            user_agent="TestBot/1.0",
        )

    def test_collector_has_correct_source_type(self, collector):
        assert collector.source_type == SourceType.REDDIT

    def test_collector_has_correct_name(self, collector):
        assert collector.name == "reddit"

    def test_collector_stores_subreddits(self, collector):
        assert "wallstreetbets" in collector.subreddits
        assert "stocks" in collector.subreddits

    def test_parse_submission_to_message(self, collector):
        mock_submission = MagicMock()
        mock_submission.id = "abc123"
        mock_submission.author.name = "trader99"
        mock_submission.title = "DD on $AAPL earnings"
        mock_submission.selftext = "Here's my analysis..."
        mock_submission.created_utc = datetime.now(timezone.utc).timestamp()
        mock_submission.permalink = "/r/stocks/comments/abc123"
        mock_submission.subreddit.display_name = "stocks"
        mock_submission.score = 150
        mock_submission.num_comments = 45

        msg = collector._parse_submission(mock_submission)

        assert msg.source == SourceType.REDDIT
        assert msg.source_id == "abc123"
        assert msg.author == "trader99"
        assert "$AAPL" in msg.content
        assert msg.subreddit == "stocks"
        assert msg.upvotes == 150
        assert msg.comment_count == 45

    def test_parse_submission_combines_title_and_body(self, collector):
        mock_submission = MagicMock()
        mock_submission.id = "xyz789"
        mock_submission.author.name = "user"
        mock_submission.title = "$NVDA breaking out"
        mock_submission.selftext = "Great volume on $NVDA today"
        mock_submission.created_utc = datetime.now(timezone.utc).timestamp()
        mock_submission.permalink = "/r/stocks/xyz789"
        mock_submission.subreddit.display_name = "stocks"
        mock_submission.score = 10
        mock_submission.num_comments = 5

        msg = collector._parse_submission(mock_submission)

        assert "$NVDA breaking out" in msg.content
        assert "Great volume" in msg.content

    def test_extract_tickers_from_reddit_post(self, collector):
        mock_submission = MagicMock()
        mock_submission.id = "test"
        mock_submission.author.name = "user"
        mock_submission.title = "$GME $AMC to the moon"
        mock_submission.selftext = "Also looking at $BB"
        mock_submission.created_utc = datetime.now(timezone.utc).timestamp()
        mock_submission.permalink = "/r/wsb/test"
        mock_submission.subreddit.display_name = "wallstreetbets"
        mock_submission.score = 1000
        mock_submission.num_comments = 500

        msg = collector._parse_submission(mock_submission)
        tickers = msg.extract_tickers()

        assert "GME" in tickers
        assert "AMC" in tickers
        assert "BB" in tickers
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/collectors/test_reddit_collector.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/collectors/reddit_collector.py
from datetime import datetime, timezone
from typing import AsyncIterator

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType


class RedditCollector(BaseCollector):
    """Collector for Reddit using asyncpraw."""

    def __init__(
        self,
        subreddits: list[str],
        client_id: str,
        client_secret: str,
        user_agent: str,
        use_streaming: bool = True,
    ):
        super().__init__(name="reddit", source_type=SourceType.REDDIT)
        self._subreddits = subreddits
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._use_streaming = use_streaming
        self._reddit = None

    @property
    def subreddits(self) -> list[str]:
        return self._subreddits

    async def connect(self) -> None:
        """Initialize asyncpraw Reddit connection."""
        try:
            import asyncpraw
            self._reddit = asyncpraw.Reddit(
                client_id=self._client_id,
                client_secret=self._client_secret,
                user_agent=self._user_agent,
            )
            self._connected = True
        except ImportError:
            raise RuntimeError("asyncpraw not installed. Run: pip install asyncpraw")

    async def disconnect(self) -> None:
        """Close Reddit connection."""
        if self._reddit:
            await self._reddit.close()
        self._reddit = None
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream submissions from subreddits.

        Uses Reddit's streaming API for real-time updates.
        """
        if not self._connected or self._reddit is None:
            raise RuntimeError("Collector not connected. Call connect() first.")

        # Combine subreddits into multi-reddit
        subreddit_str = "+".join(self._subreddits)
        subreddit = await self._reddit.subreddit(subreddit_str)

        if self._use_streaming:
            async for submission in subreddit.stream.submissions(skip_existing=True):
                yield self._parse_submission(submission)
        else:
            # Fallback to polling new posts
            async for submission in subreddit.new(limit=25):
                yield self._parse_submission(submission)

    def _parse_submission(self, submission) -> SocialMessage:
        """Convert a Reddit submission to SocialMessage."""
        # Combine title and selftext for full content
        content = submission.title
        if submission.selftext:
            content = f"{submission.title}\n\n{submission.selftext}"

        # Handle deleted authors
        author_name = "[deleted]"
        if submission.author:
            author_name = submission.author.name

        return SocialMessage(
            source=SourceType.REDDIT,
            source_id=submission.id,
            author=author_name,
            content=content,
            timestamp=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
            url=f"https://reddit.com{submission.permalink}",
            subreddit=submission.subreddit.display_name,
            upvotes=submission.score,
            comment_count=submission.num_comments,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/collectors/test_reddit_collector.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/collectors/reddit_collector.py tests/collectors/test_reddit_collector.py
git commit -m "feat(collectors): add RedditCollector with asyncpraw"
```

---

## Task 6: Stocktwits Collector

**Files:**
- Create: `src/collectors/stocktwits_collector.py`
- Test: `tests/collectors/test_stocktwits_collector.py`

**Step 1: Write the failing test**

```python
# tests/collectors/test_stocktwits_collector.py
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from src.collectors.stocktwits_collector import StocktwitsCollector
from src.models.social_message import SourceType


class TestStocktwitsCollector:
    @pytest.fixture
    def collector(self):
        return StocktwitsCollector(watchlist=["AAPL", "NVDA", "TSLA"])

    def test_collector_has_correct_source_type(self, collector):
        assert collector.source_type == SourceType.STOCKTWITS

    def test_collector_has_correct_name(self, collector):
        assert collector.name == "stocktwits"

    def test_collector_stores_watchlist(self, collector):
        assert "AAPL" in collector.watchlist
        assert "NVDA" in collector.watchlist

    def test_parse_message_to_social_message(self, collector):
        mock_msg = {
            "id": 12345,
            "body": "$NVDA looking bullish today!",
            "created_at": "2026-01-17T10:30:00Z",
            "user": {"username": "stocktrader99"},
            "entities": {
                "sentiment": {"basic": "Bullish"}
            },
        }

        msg = collector._parse_message(mock_msg, ticker="NVDA")

        assert msg.source == SourceType.STOCKTWITS
        assert msg.source_id == "12345"
        assert msg.author == "stocktrader99"
        assert "$NVDA" in msg.content
        assert msg.sentiment == "Bullish"

    def test_parse_message_no_sentiment(self, collector):
        mock_msg = {
            "id": 67890,
            "body": "What do you think about $AAPL?",
            "created_at": "2026-01-17T11:00:00Z",
            "user": {"username": "newbie"},
            "entities": {},
        }

        msg = collector._parse_message(mock_msg, ticker="AAPL")

        assert msg.sentiment is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/collectors/test_stocktwits_collector.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/collectors/stocktwits_collector.py
from datetime import datetime, timezone
from typing import AsyncIterator

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage, SourceType


class StocktwitsCollector(BaseCollector):
    """Collector for Stocktwits using pytwits."""

    def __init__(
        self,
        watchlist: list[str],
        refresh_interval: int = 30,
    ):
        super().__init__(name="stocktwits", source_type=SourceType.STOCKTWITS)
        self._watchlist = watchlist
        self._refresh_interval = refresh_interval
        self._client = None
        self._last_message_ids: dict[str, int] = {}

    @property
    def watchlist(self) -> list[str]:
        return self._watchlist

    async def connect(self) -> None:
        """Initialize pytwits client."""
        try:
            from pytwits import Streamer
            self._client = Streamer()
            self._connected = True
        except ImportError:
            raise RuntimeError("pytwits not installed. Run: pip install pytwits")

    async def disconnect(self) -> None:
        """Close client."""
        self._client = None
        self._connected = False

    async def stream(self) -> AsyncIterator[SocialMessage]:
        """Stream messages for watchlist tickers.

        Note: pytwits doesn't support true streaming, so this
        polls for new messages at the configured interval.
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Collector not connected. Call connect() first.")

        for ticker in self._watchlist:
            async for msg in self._get_ticker_messages(ticker):
                yield msg

    async def _get_ticker_messages(self, ticker: str, limit: int = 30):
        """Get recent messages for a ticker."""
        if self._client is None:
            return

        try:
            # pytwits returns a dict with 'messages' key
            response = self._client.get_symbol_msgs(ticker)
            messages = response.get("messages", [])

            for msg in messages[:limit]:
                msg_id = msg.get("id", 0)
                last_id = self._last_message_ids.get(ticker, 0)

                if msg_id > last_id:
                    self._last_message_ids[ticker] = msg_id
                    yield self._parse_message(msg, ticker)
        except Exception:
            # Log error but continue with other tickers
            pass

    def _parse_message(self, msg: dict, ticker: str) -> SocialMessage:
        """Convert a Stocktwits message to SocialMessage."""
        # Parse timestamp
        created_at = msg.get("created_at", "")
        try:
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now(timezone.utc)

        # Extract sentiment if available
        sentiment = None
        entities = msg.get("entities", {})
        if entities:
            sentiment_data = entities.get("sentiment", {})
            sentiment = sentiment_data.get("basic")

        return SocialMessage(
            source=SourceType.STOCKTWITS,
            source_id=str(msg.get("id", "")),
            author=msg.get("user", {}).get("username", "unknown"),
            content=msg.get("body", ""),
            timestamp=timestamp,
            url=f"https://stocktwits.com/symbol/{ticker}",
            sentiment=sentiment,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/collectors/test_stocktwits_collector.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/collectors/stocktwits_collector.py tests/collectors/test_stocktwits_collector.py
git commit -m "feat(collectors): add StocktwitsCollector with pytwits"
```

---

## Task 7: Alpaca Client

**Files:**
- Create: `src/execution/alpaca_client.py`
- Test: `tests/execution/test_alpaca_client.py`

**Step 1: Create execution directories**

```bash
mkdir -p src/execution tests/execution
touch src/execution/__init__.py tests/execution/__init__.py
```

**Step 2: Write the failing test**

```python
# tests/execution/test_alpaca_client.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.execution.alpaca_client import AlpacaClient


class TestAlpacaClient:
    @pytest.fixture
    def client(self):
        return AlpacaClient(
            api_key="test_key",
            secret_key="test_secret",
            paper=True,
        )

    def test_client_initializes_with_paper_mode(self, client):
        assert client.paper is True
        assert "paper" in client.base_url

    def test_client_initializes_with_live_mode(self):
        client = AlpacaClient(
            api_key="key",
            secret_key="secret",
            paper=False,
        )
        assert client.paper is False
        assert "paper" not in client.base_url

    @pytest.mark.asyncio
    async def test_get_account(self, client):
        with patch.object(client, '_trading_client') as mock_client:
            mock_account = MagicMock()
            mock_account.cash = "50000.00"
            mock_account.portfolio_value = "52000.00"
            mock_account.buying_power = "100000.00"
            mock_client.get_account.return_value = mock_account

            account = await client.get_account()

            assert account["cash"] == 50000.00
            assert account["portfolio_value"] == 52000.00

    @pytest.mark.asyncio
    async def test_get_position(self, client):
        with patch.object(client, '_trading_client') as mock_client:
            mock_position = MagicMock()
            mock_position.symbol = "NVDA"
            mock_position.qty = "100"
            mock_position.avg_entry_price = "140.50"
            mock_position.current_price = "142.00"
            mock_position.unrealized_pl = "150.00"
            mock_client.get_position.return_value = mock_position

            position = await client.get_position("NVDA")

            assert position["symbol"] == "NVDA"
            assert position["qty"] == 100
            assert position["avg_entry_price"] == 140.50

    @pytest.mark.asyncio
    async def test_submit_order(self, client):
        with patch.object(client, '_trading_client') as mock_client:
            mock_order = MagicMock()
            mock_order.id = "order_123"
            mock_order.status = "accepted"
            mock_order.symbol = "AAPL"
            mock_order.qty = "50"
            mock_order.side = "buy"
            mock_client.submit_order.return_value = mock_order

            order = await client.submit_order(
                symbol="AAPL",
                qty=50,
                side="buy",
                order_type="limit",
                limit_price=150.00,
            )

            assert order["id"] == "order_123"
            assert order["status"] == "accepted"
            assert order["symbol"] == "AAPL"
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/execution/test_alpaca_client.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Write minimal implementation**

```python
# src/execution/alpaca_client.py
from typing import Optional


class AlpacaClient:
    """Unified client for Alpaca Trading API."""

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._base_url = self.PAPER_URL if paper else self.LIVE_URL
        self._trading_client = None
        self._data_client = None

    @property
    def paper(self) -> bool:
        return self._paper

    @property
    def base_url(self) -> str:
        return self._base_url

    async def connect(self) -> None:
        """Initialize Alpaca clients."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self._trading_client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self._paper,
            )
            self._data_client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
            )
        except ImportError:
            raise RuntimeError("alpaca-py not installed. Run: pip install alpaca-py")

    async def disconnect(self) -> None:
        """Close clients."""
        self._trading_client = None
        self._data_client = None

    async def get_account(self) -> dict:
        """Get account information."""
        if not self._trading_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        account = self._trading_client.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity) if hasattr(account, 'equity') else None,
            "currency": account.currency if hasattr(account, 'currency') else "USD",
        }

    async def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a symbol."""
        if not self._trading_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        try:
            position = self._trading_client.get_position(symbol)
            return {
                "symbol": position.symbol,
                "qty": int(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc) if hasattr(position, 'unrealized_plpc') else None,
            }
        except Exception:
            return None

    async def get_all_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self._trading_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        positions = self._trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,  # "buy" or "sell"
        order_type: str = "market",  # "market", "limit", "stop", "stop_limit"
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> dict:
        """Submit an order."""
        if not self._trading_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif_enum = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC

        if order_type == "market":
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif_enum,
            )
        elif order_type == "limit" and limit_price:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=tif_enum,
                limit_price=limit_price,
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        order = self._trading_client.submit_order(request)
        return {
            "id": str(order.id),
            "status": str(order.status.value) if hasattr(order.status, 'value') else str(order.status),
            "symbol": order.symbol,
            "qty": int(order.qty) if order.qty else qty,
            "side": side,
            "order_type": order_type,
        }

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self._trading_client:
            raise RuntimeError("Client not connected. Call connect() first.")

        try:
            self._trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/execution/test_alpaca_client.py -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add src/execution/alpaca_client.py tests/execution/test_alpaca_client.py src/execution/__init__.py tests/execution/__init__.py
git commit -m "feat(execution): add AlpacaClient for trading operations"
```

---

## Task 8: Collector Manager

**Files:**
- Create: `src/collectors/collector_manager.py`
- Test: `tests/collectors/test_collector_manager.py`

**Step 1: Write the failing test**

```python
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
        assert len(manager.collectors) == 2

    @pytest.mark.asyncio
    async def test_manager_connects_all(self, manager):
        await manager.connect_all()
        for collector in manager.collectors:
            assert collector.is_connected

    @pytest.mark.asyncio
    async def test_manager_disconnects_all(self, manager):
        await manager.connect_all()
        await manager.disconnect_all()
        for collector in manager.collectors:
            assert not collector.is_connected

    @pytest.mark.asyncio
    async def test_manager_collects_from_all(self, manager, messages):
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

    def test_add_callback(self, manager):
        callback = MagicMock()
        manager.add_callback(callback)
        assert callback in manager.callbacks
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/collectors/test_collector_manager.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# src/collectors/collector_manager.py
import asyncio
from typing import AsyncIterator, Callable

from src.collectors.base import BaseCollector
from src.models.social_message import SocialMessage


class CollectorManager:
    """Manages multiple collectors and aggregates their streams."""

    def __init__(self, collectors: list[BaseCollector]):
        self._collectors = collectors
        self._callbacks: list[Callable[[SocialMessage], None]] = []
        self._running = False

    @property
    def collectors(self) -> list[BaseCollector]:
        return self._collectors

    @property
    def callbacks(self) -> list[Callable[[SocialMessage], None]]:
        return self._callbacks

    def add_callback(self, callback: Callable[[SocialMessage], None]) -> None:
        """Add a callback to be called for each message."""
        self._callbacks.append(callback)

    async def connect_all(self) -> None:
        """Connect all collectors."""
        await asyncio.gather(*[c.connect() for c in self._collectors])

    async def disconnect_all(self) -> None:
        """Disconnect all collectors."""
        self._running = False
        await asyncio.gather(*[c.disconnect() for c in self._collectors])

    async def stream_all(self) -> AsyncIterator[SocialMessage]:
        """Stream messages from all collectors.

        Uses asyncio.Queue to merge streams from all collectors.
        """
        self._running = True
        queue: asyncio.Queue[SocialMessage] = asyncio.Queue()

        async def collector_worker(collector: BaseCollector):
            try:
                async for msg in collector.stream():
                    if not self._running:
                        break
                    await queue.put(msg)
                    # Call callbacks
                    for callback in self._callbacks:
                        try:
                            callback(msg)
                        except Exception:
                            pass
            except Exception:
                pass

        # Start all collector workers
        tasks = [
            asyncio.create_task(collector_worker(c))
            for c in self._collectors
        ]

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield msg
                except asyncio.TimeoutError:
                    # Check if all tasks are done
                    if all(t.done() for t in tasks):
                        break
        finally:
            # Clean up tasks
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self) -> None:
        """Run the collector manager continuously."""
        await self.connect_all()
        try:
            async for msg in self.stream_all():
                # Messages are processed via callbacks
                pass
        finally:
            await self.disconnect_all()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/collectors/test_collector_manager.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/collectors/collector_manager.py tests/collectors/test_collector_manager.py
git commit -m "feat(collectors): add CollectorManager to aggregate streams"
```

---

## Task 9: Main Entry Point

**Files:**
- Create: `main.py`
- Create: `config/settings.yaml`

**Step 1: Create main.py**

```python
# main.py
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from src.config.settings import Settings
from src.collectors.twitter_collector import TwitterCollector
from src.collectors.reddit_collector import RedditCollector
from src.collectors.stocktwits_collector import StocktwitsCollector
from src.collectors.collector_manager import CollectorManager
from src.execution.alpaca_client import AlpacaClient


async def main():
    """Main entry point for the trading system."""
    # Load environment variables
    load_dotenv()

    # Load settings
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        print("Error: config/settings.yaml not found")
        return

    settings = Settings.from_yaml(config_path)
    print(f"Starting {settings.system.name} in {settings.system.mode} mode")

    # Initialize collectors
    collectors = []

    if settings.collectors.twitter.enabled:
        twitter = TwitterCollector(
            accounts_to_follow=["unusual_whales", "FirstSquawk"],
            refresh_interval=settings.collectors.twitter.refresh_interval_seconds,
        )
        collectors.append(twitter)
        print("Twitter collector enabled")

    if settings.collectors.reddit.enabled:
        reddit = RedditCollector(
            subreddits=["wallstreetbets", "stocks"],
            client_id=settings.reddit_api.client_id,
            client_secret=settings.reddit_api.client_secret,
            user_agent=settings.reddit_api.user_agent,
            use_streaming=settings.collectors.reddit.use_streaming,
        )
        collectors.append(reddit)
        print("Reddit collector enabled")

    if settings.collectors.stocktwits.enabled:
        stocktwits = StocktwitsCollector(
            watchlist=["AAPL", "NVDA", "TSLA", "AMD", "META"],
            refresh_interval=settings.collectors.stocktwits.refresh_interval_seconds,
        )
        collectors.append(stocktwits)
        print("Stocktwits collector enabled")

    # Initialize Alpaca client
    alpaca = AlpacaClient(
        api_key=settings.alpaca.api_key,
        secret_key=settings.alpaca.secret_key,
        paper=settings.alpaca.paper,
    )
    print(f"Alpaca client initialized (paper={settings.alpaca.paper})")

    # Create collector manager
    manager = CollectorManager(collectors=collectors)

    # Add message callback
    def on_message(msg):
        tickers = msg.extract_tickers()
        if tickers:
            print(f"[{msg.source.value}] @{msg.author}: {tickers}")

    manager.add_callback(on_message)

    # Connect and run
    try:
        await alpaca.connect()
        account = await alpaca.get_account()
        print(f"Account cash: ${account['cash']:,.2f}")

        print("\nStarting collectors...")
        await manager.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await manager.disconnect_all()
        await alpaca.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Create config/settings.yaml**

```yaml
# config/settings.yaml
system:
  name: "Intraday Trading System"
  version: "1.0.0"
  mode: "paper"
  timezone: "America/New_York"

collectors:
  twitter:
    enabled: true
    engine: "twscrape"
    refresh_interval_seconds: 15

  reddit:
    enabled: true
    use_streaming: true

  stocktwits:
    enabled: true
    refresh_interval_seconds: 30

risk:
  circuit_breakers:
    per_trade:
      max_loss_percent: 1.0
    daily:
      max_loss_percent: 3.0
    weekly:
      max_loss_percent: 6.0
```

**Step 3: Run to verify (dry run)**

```bash
python -c "from src.config.settings import Settings; print('Import OK')"
```

**Step 4: Commit**

```bash
git add main.py config/settings.yaml
git commit -m "feat: add main entry point with collector orchestration"
```

---

## Task 10: Run All Tests

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass (approximately 30+ tests)

**Step 2: Check coverage**

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: Phase 1 Core implementation complete"
```

---

## Summary

Phase 1 delivers:

| Component | Status |
|-----------|--------|
| SocialMessage model | ✅ |
| BaseCollector interface | ✅ |
| TwitterCollector | ✅ |
| RedditCollector | ✅ |
| StocktwitsCollector | ✅ |
| CollectorManager | ✅ |
| AlpacaClient | ✅ |
| Configuration system | ✅ |
| Main entry point | ✅ |

**Next Phase:** Phase 2 - Analysis (FinTwitBERT sentiment, Claude integration, technical validation)
