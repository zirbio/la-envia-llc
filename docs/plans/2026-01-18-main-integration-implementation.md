# Main.py Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate all 7-layer trading pipeline components into main.py with fail-fast validation, graceful shutdown, and full observability.

**Architecture:** Sequential initialization of components (Config → Infrastructure → Analysis → Collectors → Pipeline → Orchestrator) with shared AlpacaClient dependency injection. TradingOrchestrator coordinates message flow through the complete pipeline.

**Tech Stack:** Python 3.13, asyncio, Pydantic settings, transformers (FinTwitBERT), Anthropic Claude API, Alpaca API, Telegram Bot API

---

## Task 1: Helper Functions & Data Directory Setup

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`

**Step 1: Write tests for helper functions**

Create `tests/test_main.py`:

```python
"""Tests for main.py helper functions."""
import os
from pathlib import Path
from unittest.mock import patch
import pytest


def test_validate_env_vars_success():
    """Test env var validation with all required vars present."""
    from main import validate_env_vars

    with patch.dict(os.environ, {
        "ALPACA_API_KEY": "test_key",
        "ALPACA_SECRET_KEY": "test_secret",
        "ANTHROPIC_API_KEY": "test_anthropic",
    }):
        # Should not raise
        validate_env_vars()


def test_validate_env_vars_missing_alpaca_key():
    """Test env var validation fails when ALPACA_API_KEY missing."""
    from main import validate_env_vars

    with patch.dict(os.environ, {
        "ALPACA_SECRET_KEY": "test_secret",
        "ANTHROPIC_API_KEY": "test_anthropic",
    }, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            validate_env_vars()
        assert "ALPACA_API_KEY" in str(exc_info.value)


def test_create_data_dirs(tmp_path):
    """Test data directory creation."""
    from main import create_data_dirs

    with patch("main.Path", return_value=tmp_path):
        create_data_dirs()

        assert (tmp_path / "data" / "trades").exists()
        assert (tmp_path / "data" / "signals").exists()
        assert (tmp_path / "data" / "cache").exists()
        assert (tmp_path / "data" / "backtest_results").exists()
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/silvio_requena/Code/la\ envia\ llc/.worktrees/main-integration
uv run pytest tests/test_main.py -v
```

Expected: FAIL with "cannot import name 'validate_env_vars'"

**Step 3: Implement helper functions in main.py**

Add to top of `main.py` after imports:

```python
import sys
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_env_vars() -> None:
    """Validate required environment variables are set.

    Raises:
        SystemExit: If any required env var is missing.
    """
    required_vars = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "ANTHROPIC_API_KEY",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please check your .env file")
        sys.exit(1)


def create_data_dirs() -> None:
    """Create required data directories if they don't exist."""
    dirs = [
        Path("data/trades"),
        Path("data/signals"),
        Path("data/cache"),
        Path("data/backtest_results"),
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Data directories verified")


def print_startup_banner(settings) -> None:
    """Print system startup banner."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.system.name}")
    logger.info(f"Mode: {settings.system.mode}")
    logger.info(f"Version: {settings.system.version}")
    logger.info("=" * 60)
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_main.py -v
```

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add helper functions for env validation and data dirs

- validate_env_vars() checks required env vars
- create_data_dirs() creates data directory structure
- print_startup_banner() displays system info
- Add logging configuration

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Phase 1 - Configuration Loading & Validation

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main.py`

**Step 1: Write test for configuration phase**

Add to `tests/test_main.py`:

```python
def test_load_and_validate_config_success(tmp_path):
    """Test successful config loading and validation."""
    from main import load_and_validate_config

    # Create temporary .env
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY=test\n"
        "ALPACA_SECRET_KEY=test\n"
        "ANTHROPIC_API_KEY=test\n"
    )

    # Create temporary settings.yaml
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("system:\n  name: Test\n  mode: paper\n")

    with patch("main.load_dotenv"):
        with patch("main.Path") as mock_path:
            mock_path.return_value = config_file
            settings = load_and_validate_config()

    assert settings is not None


def test_load_and_validate_config_missing_yaml():
    """Test config loading fails with missing YAML."""
    from main import load_and_validate_config

    with patch("main.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        with pytest.raises(SystemExit):
            load_and_validate_config()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_main.py::test_load_and_validate_config_success -v
```

Expected: FAIL with "cannot import name 'load_and_validate_config'"

**Step 3: Implement configuration loading**

Add to `main.py` after helper functions:

```python
def load_and_validate_config() -> Settings:
    """Load and validate configuration.

    Returns:
        Settings object loaded from YAML.

    Raises:
        SystemExit: If config file missing or env vars invalid.
    """
    # Load environment variables
    load_dotenv()
    logger.info("✓ Loaded environment variables")

    # Validate env vars
    validate_env_vars()
    logger.info("✓ Environment variables validated")

    # Check settings.yaml exists
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        logger.error("config/settings.yaml not found")
        sys.exit(1)

    # Load settings
    settings = Settings.from_yaml(config_path)
    logger.info("✓ Settings loaded from config/settings.yaml")

    # Create data directories
    create_data_dirs()

    return settings
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_main.py::test_load_and_validate_config_success -v
uv run pytest tests/test_main.py::test_load_and_validate_config_missing_yaml -v
```

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add config loading and validation phase

- load_and_validate_config() orchestrates Phase 1
- Validates env vars before loading settings
- Creates data directories
- Fail-fast on missing config

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Phase 2 - Core Infrastructure (Alpaca + Telegram)

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main.py`

**Step 1: Write test for infrastructure initialization**

Add to `tests/test_main.py`:

```python
@pytest.mark.asyncio
async def test_initialize_infrastructure_success():
    """Test successful infrastructure initialization."""
    from main import initialize_infrastructure
    from src.config.settings import Settings

    settings = Settings.from_yaml(Path("config/settings.yaml"))

    with patch("main.AlpacaClient") as mock_alpaca:
        with patch("main.TelegramNotifier") as mock_telegram:
            with patch("main.AlertFormatter"):
                # Mock successful connection
                mock_alpaca_instance = mock_alpaca.return_value
                mock_alpaca_instance.connect = AsyncMock()
                mock_alpaca_instance.get_account = AsyncMock(
                    return_value={"cash": "100000.00"}
                )

                mock_telegram_instance = mock_telegram.return_value
                mock_telegram_instance.start = AsyncMock()
                mock_telegram_instance.send_alert = AsyncMock()

                alpaca, telegram = await initialize_infrastructure(settings)

                assert alpaca is not None
                assert telegram is not None
                mock_alpaca_instance.connect.assert_called_once()
                mock_telegram_instance.start.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_main.py::test_initialize_infrastructure_success -v
```

Expected: FAIL with "cannot import name 'initialize_infrastructure'"

**Step 3: Add imports and implement infrastructure initialization**

Add to imports section in `main.py`:

```python
from src.notifications import TelegramNotifier, AlertFormatter
```

Add function after `load_and_validate_config()`:

```python
async def initialize_infrastructure(settings: Settings) -> tuple[AlpacaClient, TelegramNotifier]:
    """Initialize core infrastructure (Alpaca + Telegram).

    Args:
        settings: Loaded settings object.

    Returns:
        Tuple of (AlpacaClient, TelegramNotifier).

    Raises:
        SystemExit: If connection fails.
    """
    # Initialize Alpaca
    try:
        alpaca_client = AlpacaClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            paper=settings.alpaca.paper,
        )
        await alpaca_client.connect()

        account = await alpaca_client.get_account()
        cash = float(account["cash"])
        logger.info(f"✓ Alpaca connected (Paper mode: {settings.alpaca.paper}, Cash: ${cash:,.2f})")

    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        logger.error("Check ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        sys.exit(1)

    # Initialize Telegram
    try:
        formatter = AlertFormatter()
        telegram = TelegramNotifier(settings=settings.notifications, formatter=formatter)
        await telegram.start()
        await telegram.send_alert("🚀 System starting up...")
        logger.info("✓ Telegram connected (sent startup message)")

    except Exception as e:
        logger.error(f"Failed to connect to Telegram: {e}")
        logger.error("Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        sys.exit(1)

    return alpaca_client, telegram
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_main.py::test_initialize_infrastructure_success -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add Phase 2 infrastructure initialization

- Initialize and connect AlpacaClient
- Initialize and test TelegramNotifier
- Fail-fast on connection errors
- Send startup notification to Telegram

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Phase 3 - Analysis Components

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main.py`

**Step 1: Write test for analysis components**

Add to `tests/test_main.py`:

```python
@pytest.mark.asyncio
async def test_initialize_analyzers_success():
    """Test successful analyzer initialization."""
    from main import initialize_analyzers
    from src.config.settings import Settings

    settings = Settings.from_yaml(Path("config/settings.yaml"))

    with patch("main.SentimentAnalyzer") as mock_sent:
        with patch("main.ClaudeAnalyzer") as mock_claude:
            with patch("main.AnalyzerManager") as mock_manager:
                # Mock successful init
                mock_sent_instance = mock_sent.return_value
                mock_claude_instance = mock_claude.return_value
                mock_claude_instance.test_connection = AsyncMock()

                analyzer_manager = await initialize_analyzers(settings)

                assert analyzer_manager is not None
                mock_claude_instance.test_connection.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_main.py::test_initialize_analyzers_success -v
```

Expected: FAIL with "cannot import name 'initialize_analyzers'"

**Step 3: Add imports and implement analyzer initialization**

Add to imports in `main.py`:

```python
from src.analyzers import AnalyzerManager, SentimentAnalyzer, ClaudeAnalyzer
```

Add function:

```python
async def initialize_analyzers(settings: Settings) -> AnalyzerManager:
    """Initialize analysis components (FinTwitBERT + Claude).

    Args:
        settings: Loaded settings object.

    Returns:
        AnalyzerManager instance.

    Raises:
        SystemExit: If model download or API connection fails.
    """
    # Initialize SentimentAnalyzer
    try:
        sentiment_analyzer = SentimentAnalyzer(
            model_name=settings.analyzers.sentiment.model,
            batch_size=settings.analyzers.sentiment.batch_size,
            min_confidence=settings.analyzers.sentiment.min_confidence,
        )
        logger.info("✓ FinTwitBERT model loaded")

    except Exception as e:
        logger.error(f"Failed to load FinTwitBERT model: {e}")
        logger.error("Check internet connection for model download")
        sys.exit(1)

    # Initialize ClaudeAnalyzer
    try:
        claude_analyzer = ClaudeAnalyzer(
            model=settings.analyzers.claude.model,
            max_tokens=settings.analyzers.claude.max_tokens,
        )
        # Test connection
        await claude_analyzer.test_connection()
        logger.info("✓ Claude API verified")

    except Exception as e:
        logger.error(f"Claude API test failed: {e}")
        logger.error("Check ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    # Create AnalyzerManager
    analyzer_manager = AnalyzerManager(
        sentiment_analyzer=sentiment_analyzer,
        claude_analyzer=claude_analyzer,
        min_sentiment_confidence=settings.analyzers.sentiment.min_confidence,
        enable_deep_analysis=settings.analyzers.claude.enabled,
    )

    return analyzer_manager
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_main.py::test_initialize_analyzers_success -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add Phase 3 analyzer initialization

- Initialize SentimentAnalyzer with FinTwitBERT
- Initialize ClaudeAnalyzer and test API connection
- Create AnalyzerManager combining both
- Fail-fast on model/API errors

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Phase 4 - Collectors

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main.py`

**Step 1: Write test for collectors**

Add to `tests/test_main.py`:

```python
def test_initialize_collectors_success():
    """Test successful collector initialization."""
    from main import initialize_collectors
    from src.config.settings import Settings
    from src.execution.alpaca_client import AlpacaClient

    settings = Settings.from_yaml(Path("config/settings.yaml"))
    mock_alpaca = MagicMock(spec=AlpacaClient)

    with patch("main.StocktwitsCollector") as mock_st:
        with patch("main.RedditCollector") as mock_reddit:
            with patch("main.CollectorManager") as mock_manager:
                with patch.dict(os.environ, {}, clear=True):  # No Reddit creds
                    collector_manager = initialize_collectors(settings, mock_alpaca)

                    assert collector_manager is not None
                    # Should create Stocktwits (no auth needed)
                    mock_st.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_main.py::test_initialize_collectors_success -v
```

Expected: FAIL with "cannot import name 'initialize_collectors'"

**Step 3: Add imports and implement collector initialization**

Add to imports:

```python
from src.collectors import CollectorManager, StocktwitsCollector, RedditCollector
```

Add function:

```python
def initialize_collectors(settings: Settings, alpaca_client: AlpacaClient) -> CollectorManager:
    """Initialize data collectors.

    Args:
        settings: Loaded settings object.
        alpaca_client: Connected Alpaca client for news.

    Returns:
        CollectorManager with available collectors.

    Raises:
        SystemExit: If no collectors available.
    """
    collectors = []

    # Stocktwits (no auth required)
    if settings.collectors.stocktwits.enabled:
        stocktwits = StocktwitsCollector(
            watchlist=["AAPL", "NVDA", "TSLA", "AMD", "META"],
            refresh_interval=settings.collectors.stocktwits.refresh_interval_seconds,
        )
        collectors.append(stocktwits)
        logger.info("✓ Stocktwits collector initialized")

    # Alpaca News (uses existing client)
    # TODO: Add AlpacaNewsCollector when implemented

    # Reddit (optional - skip if no credentials)
    if settings.collectors.reddit.enabled:
        reddit_id = os.getenv("REDDIT_CLIENT_ID")
        reddit_secret = os.getenv("REDDIT_CLIENT_SECRET")

        if reddit_id and reddit_secret:
            try:
                reddit = RedditCollector(
                    subreddits=["wallstreetbets", "stocks", "options"],
                    client_id=reddit_id,
                    client_secret=reddit_secret,
                    user_agent=settings.reddit_api.user_agent,
                    use_streaming=settings.collectors.reddit.use_streaming,
                )
                collectors.append(reddit)
                logger.info("✓ Reddit collector initialized")
            except Exception as e:
                logger.warning(f"Reddit collector init failed: {e}")
                logger.warning("Continuing without Reddit")
        else:
            logger.warning("Reddit credentials not found - skipping Reddit collector")
            logger.warning("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to enable")

    # Verify we have at least one collector
    if not collectors:
        logger.error("No collectors available")
        logger.error("At least Stocktwits should work")
        sys.exit(1)

    collector_names = [c.__class__.__name__ for c in collectors]
    logger.info(f"✓ Collectors initialized: {', '.join(collector_names)} ({len(collectors)}/{3})")

    return CollectorManager(collectors=collectors)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_main.py::test_initialize_collectors_success -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add Phase 4 collector initialization

- Initialize StocktwitsCollector (always available)
- Initialize RedditCollector if credentials present
- Graceful skip if Reddit creds missing (warning, not error)
- Fail-fast if no collectors available

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Phase 5 - Pipeline Components (SignalScorer, Validator, Gate, Risk, Journal, Executor)

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main.py`

**Step 1: Write test for pipeline components**

Add to `tests/test_main.py`:

```python
def test_initialize_pipeline_components_success():
    """Test successful pipeline component initialization."""
    from main import initialize_pipeline_components
    from src.config.settings import Settings
    from src.execution.alpaca_client import AlpacaClient

    settings = Settings.from_yaml(Path("config/settings.yaml"))
    mock_alpaca = MagicMock(spec=AlpacaClient)

    components = initialize_pipeline_components(settings, mock_alpaca)

    assert "scorer" in components
    assert "validator" in components
    assert "gate" in components
    assert "risk_manager" in components
    assert "journal" in components
    assert "executor" in components
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_main.py::test_initialize_pipeline_components_success -v
```

Expected: FAIL with "cannot import name 'initialize_pipeline_components'"

**Step 3: Add imports and implement pipeline initialization**

Add to imports:

```python
from src.validators import TechnicalValidator
from src.scoring import (
    SignalScorer,
    SourceCredibilityManager,
    TimeFactorCalculator,
    ConfluenceDetector,
    DynamicWeightCalculator,
    RecommendationBuilder,
)
from src.gate import MarketGate
from src.risk import RiskManager
from src.journal import JournalManager
from src.execution import TradeExecutor
```

Add function:

```python
def initialize_pipeline_components(
    settings: Settings,
    alpaca_client: AlpacaClient,
) -> dict:
    """Initialize pipeline components (Layers 3-6).

    Args:
        settings: Loaded settings object.
        alpaca_client: Connected Alpaca client.

    Returns:
        Dict with: scorer, validator, gate, risk_manager, journal, executor.
    """
    # Layer 4: SignalScorer (requires 5 sub-components)
    credibility_manager = SourceCredibilityManager(
        tier1_sources=settings.scoring.tier1_sources,
        tier1_multiplier=settings.scoring.credibility_tier1_multiplier,
        tier2_multiplier=settings.scoring.credibility_tier2_multiplier,
        tier3_multiplier=settings.scoring.credibility_tier3_multiplier,
    )

    time_calculator = TimeFactorCalculator(
        timezone=settings.system.timezone,
        premarket_factor=settings.scoring.premarket_factor,
        afterhours_factor=settings.scoring.afterhours_factor,
        earnings_factor=settings.scoring.earnings_factor,
        earnings_proximity_days=settings.scoring.earnings_proximity_days,
    )

    confluence_detector = ConfluenceDetector(
        window_minutes=settings.scoring.confluence_window_minutes,
        bonus_2_signals=settings.scoring.confluence_bonus_2_signals,
        bonus_3_signals=settings.scoring.confluence_bonus_3_signals,
    )

    weight_calculator = DynamicWeightCalculator(
        base_sentiment_weight=settings.scoring.base_sentiment_weight,
        base_technical_weight=settings.scoring.base_technical_weight,
        strong_trend_adx=settings.scoring.strong_trend_adx,
        weak_trend_adx=settings.scoring.weak_trend_adx,
    )

    recommendation_builder = RecommendationBuilder(
        default_stop_loss_percent=settings.scoring.default_stop_loss_percent,
        default_risk_reward_ratio=settings.scoring.default_risk_reward_ratio,
        tier_strong_threshold=settings.scoring.tier_strong_threshold,
        tier_moderate_threshold=settings.scoring.tier_moderate_threshold,
        tier_weak_threshold=settings.scoring.tier_weak_threshold,
        position_size_strong=settings.scoring.position_size_strong,
        position_size_moderate=settings.scoring.position_size_moderate,
        position_size_weak=settings.scoring.position_size_weak,
    )

    signal_scorer = SignalScorer(
        credibility_manager=credibility_manager,
        time_calculator=time_calculator,
        confluence_detector=confluence_detector,
        weight_calculator=weight_calculator,
        recommendation_builder=recommendation_builder,
    )
    logger.info("✓ SignalScorer initialized (5 sub-components)")

    # Layer 3: TechnicalValidator
    technical_validator = TechnicalValidator(
        alpaca_client=alpaca_client,
        veto_mode=settings.validators.technical.veto_mode,
        rsi_overbought=settings.validators.technical.rsi_overbought,
        rsi_oversold=settings.validators.technical.rsi_oversold,
        adx_trend_threshold=settings.validators.technical.adx_trend_threshold,
        lookback_bars=settings.validators.technical.lookback_bars,
        timeframe=settings.validators.technical.timeframe,
        options_volume_spike_ratio=settings.validators.technical.options_volume_spike_ratio,
        iv_rank_warning_threshold=settings.validators.technical.iv_rank_warning_threshold,
    )
    logger.info("✓ TechnicalValidator initialized")

    # Layer 0: MarketGate
    market_gate = MarketGate(
        alpaca_client=alpaca_client,
        settings=settings.market_gate,
    )
    logger.info("✓ MarketGate initialized")

    # Layer 5: RiskManager
    risk_manager = RiskManager(
        max_position_value=settings.risk_settings.max_position_value,
        max_daily_loss=settings.risk_settings.max_daily_loss,
        unrealized_warning_threshold=settings.risk_settings.unrealized_warning_threshold,
    )
    logger.info("✓ RiskManager initialized")

    # Observability: JournalManager
    journal_manager = JournalManager(settings=settings.journal)
    logger.info("✓ JournalManager initialized")

    # Layer 6: TradeExecutor
    trade_executor = TradeExecutor(
        alpaca_client=alpaca_client,
        risk_manager=risk_manager,
    )
    logger.info("✓ TradeExecutor initialized")

    return {
        "scorer": signal_scorer,
        "validator": technical_validator,
        "gate": market_gate,
        "risk_manager": risk_manager,
        "journal": journal_manager,
        "executor": trade_executor,
    }
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_main.py::test_initialize_pipeline_components_success -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add Phase 5 pipeline component initialization

- Initialize SignalScorer with 5 sub-components
- Initialize TechnicalValidator
- Initialize MarketGate
- Initialize RiskManager
- Initialize JournalManager
- Initialize TradeExecutor
- All components configured from settings.yaml

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Phase 6 - Orchestrator

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main.py`

**Step 1: Write test for orchestrator**

Add to `tests/test_main.py`:

```python
def test_initialize_orchestrator_success():
    """Test successful orchestrator initialization."""
    from main import initialize_orchestrator
    from src.config.settings import Settings

    settings = Settings.from_yaml(Path("config/settings.yaml"))

    # Create mocks for all dependencies
    mock_collector_manager = MagicMock()
    mock_analyzer_manager = MagicMock()
    mock_components = {
        "scorer": MagicMock(),
        "validator": MagicMock(),
        "gate": MagicMock(),
        "risk_manager": MagicMock(),
        "journal": MagicMock(),
        "executor": MagicMock(),
    }

    with patch("main.TradingOrchestrator") as mock_orch:
        orchestrator = initialize_orchestrator(
            settings,
            mock_collector_manager,
            mock_analyzer_manager,
            mock_components,
        )

        assert orchestrator is not None
        mock_orch.assert_called_once()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_main.py::test_initialize_orchestrator_success -v
```

Expected: FAIL with "cannot import name 'initialize_orchestrator'"

**Step 3: Add import and implement orchestrator initialization**

Add to imports:

```python
from src.orchestrator import TradingOrchestrator
```

Add function:

```python
def initialize_orchestrator(
    settings: Settings,
    collector_manager: CollectorManager,
    analyzer_manager: AnalyzerManager,
    components: dict,
) -> TradingOrchestrator:
    """Initialize TradingOrchestrator.

    Args:
        settings: Loaded settings object.
        collector_manager: Initialized collector manager.
        analyzer_manager: Initialized analyzer manager.
        components: Dict with pipeline components.

    Returns:
        TradingOrchestrator instance.
    """
    orchestrator = TradingOrchestrator(
        collector_manager=collector_manager,
        analyzer_manager=analyzer_manager,
        technical_validator=components["validator"],
        signal_scorer=components["scorer"],
        risk_manager=components["risk_manager"],
        market_gate=components["gate"],
        trade_executor=components["executor"],
        settings=settings.orchestrator,
    )
    logger.info("✓ TradingOrchestrator initialized")

    return orchestrator
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_main.py::test_initialize_orchestrator_success -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add Phase 6 orchestrator initialization

- Initialize TradingOrchestrator with all components
- Wire collector_manager, analyzer_manager, and pipeline
- Configure from settings.orchestrator

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Rewrite main() Function with All Phases

**Files:**
- Modify: `main.py`

**Step 1: No test needed (integration test in Task 9)**

**Step 2: Rewrite main() to use all initialization functions**

Replace the existing `main()` function:

```python
async def main():
    """Main entry point for the trading system."""
    # Phase 1: Configuration
    settings = load_and_validate_config()
    print_startup_banner(settings)

    # Phase 2: Core Infrastructure
    alpaca_client, telegram = await initialize_infrastructure(settings)

    # Phase 3: Analysis Components
    analyzer_manager = await initialize_analyzers(settings)

    # Phase 4: Collectors
    collector_manager = initialize_collectors(settings, alpaca_client)

    # Phase 5: Pipeline Components
    components = initialize_pipeline_components(settings, alpaca_client)

    # Phase 6: Orchestrator
    orchestrator = initialize_orchestrator(
        settings,
        collector_manager,
        analyzer_manager,
        components,
    )

    # Start system
    logger.info("=" * 60)
    logger.info("🚀 All components initialized successfully")
    logger.info("Starting TradingOrchestrator...")
    logger.info("=" * 60)

    try:
        await orchestrator.start()
        logger.info("✓ TradingOrchestrator started")
        logger.info("\nListening for signals... (Press Ctrl+C to stop)")

        # Keep running until interrupted
        while orchestrator.is_running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Shutdown requested...")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

    finally:
        # Graceful shutdown
        logger.info("Shutting down gracefully...")

        # Stop orchestrator (processes remaining messages)
        await orchestrator.stop()
        logger.info("✓ Orchestrator stopped")

        # Disconnect infrastructure
        await alpaca_client.disconnect()
        logger.info("✓ Alpaca disconnected")

        # Final notification
        try:
            await telegram.send_alert("🛑 System shutdown complete")
            logger.info("✓ Shutdown notification sent")
        except Exception:
            pass  # Ignore telegram errors during shutdown

        logger.info("=" * 60)
        logger.info("✓ Shutdown complete")
        logger.info("=" * 60)
```

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: rewrite main() with complete 6-phase initialization

- Sequential initialization: Config → Infrastructure → Analysis → Collectors → Pipeline → Orchestrator
- Graceful shutdown: Stop orchestrator → Disconnect Alpaca → Send notification
- Full observability with logging at each phase
- Fail-fast error handling throughout

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Integration Test & Manual Verification

**Files:**
- Create: `tests/integration/test_main_startup.py`

**Step 1: Write integration test**

Create `tests/integration/test_main_startup.py`:

```python
"""Integration test for main.py startup sequence."""
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import os


@pytest.mark.asyncio
async def test_full_startup_sequence():
    """Test complete startup sequence with all phases."""
    # Mock all external dependencies
    with patch("main.load_dotenv"):
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "test_key",
            "ALPACA_SECRET_KEY": "test_secret",
            "ANTHROPIC_API_KEY": "test_anthropic",
            "TELEGRAM_BOT_TOKEN": "test_token",
            "TELEGRAM_CHAT_ID": "123456",
        }):
            with patch("main.Settings.from_yaml") as mock_settings:
                with patch("main.AlpacaClient") as mock_alpaca:
                    with patch("main.TelegramNotifier") as mock_telegram:
                        with patch("main.SentimentAnalyzer"):
                            with patch("main.ClaudeAnalyzer") as mock_claude:
                                with patch("main.TradingOrchestrator") as mock_orch:
                                    # Setup mocks
                                    mock_settings.return_value = MagicMock()

                                    alpaca_instance = mock_alpaca.return_value
                                    alpaca_instance.connect = AsyncMock()
                                    alpaca_instance.get_account = AsyncMock(
                                        return_value={"cash": "100000.00"}
                                    )
                                    alpaca_instance.disconnect = AsyncMock()

                                    telegram_instance = mock_telegram.return_value
                                    telegram_instance.start = AsyncMock()
                                    telegram_instance.send_alert = AsyncMock()

                                    claude_instance = mock_claude.return_value
                                    claude_instance.test_connection = AsyncMock()

                                    orch_instance = mock_orch.return_value
                                    orch_instance.start = AsyncMock()
                                    orch_instance.stop = AsyncMock()
                                    orch_instance.is_running = False  # Exit immediately

                                    # Run main
                                    from main import main
                                    await main()

                                    # Verify all phases executed
                                    alpaca_instance.connect.assert_called_once()
                                    telegram_instance.start.assert_called_once()
                                    claude_instance.test_connection.assert_called_once()
                                    orch_instance.start.assert_called_once()
                                    orch_instance.stop.assert_called_once()
                                    alpaca_instance.disconnect.assert_called_once()
```

**Step 2: Run integration test**

```bash
uv run pytest tests/integration/test_main_startup.py -v
```

Expected: PASS

**Step 3: Manual verification (dry run)**

Create data directories:

```bash
mkdir -p data/trades data/signals data/cache data/backtest_results
```

Run the system (will attempt real connections):

```bash
uv run python main.py
```

Expected output:
```
[2026-01-18 XX:XX:XX] INFO: ============================================================
[2026-01-18 XX:XX:XX] INFO: Starting Intraday Trading System
[2026-01-18 XX:XX:XX] INFO: Mode: paper
[2026-01-18 XX:XX:XX] INFO: Version: 1.0.0
[2026-01-18 XX:XX:XX] INFO: ============================================================
[2026-01-18 XX:XX:XX] INFO: ✓ Loaded environment variables
[2026-01-18 XX:XX:XX] INFO: ✓ Environment variables validated
[2026-01-18 XX:XX:XX] INFO: ✓ Settings loaded from config/settings.yaml
[2026-01-18 XX:XX:XX] INFO: Data directories verified
[2026-01-18 XX:XX:XX] INFO: ✓ Alpaca connected (Paper mode: True, Cash: $100,000.00)
[2026-01-18 XX:XX:XX] INFO: ✓ Telegram connected (sent startup message)
[2026-01-18 XX:XX:XX] INFO: ✓ FinTwitBERT model loaded
[2026-01-18 XX:XX:XX] INFO: ✓ Claude API verified
[2026-01-18 XX:XX:XX] INFO: ✓ Stocktwits collector initialized
[2026-01-18 XX:XX:XX] INFO: ✓ Collectors initialized: StocktwitsCollector (1/3)
[2026-01-18 XX:XX:XX] INFO: ✓ SignalScorer initialized (5 sub-components)
[2026-01-18 XX:XX:XX] INFO: ✓ TechnicalValidator initialized
[2026-01-18 XX:XX:XX] INFO: ✓ MarketGate initialized
[2026-01-18 XX:XX:XX] INFO: ✓ RiskManager initialized
[2026-01-18 XX:XX:XX] INFO: ✓ JournalManager initialized
[2026-01-18 XX:XX:XX] INFO: ✓ TradeExecutor initialized
[2026-01-18 XX:XX:XX] INFO: ✓ TradingOrchestrator initialized
[2026-01-18 XX:XX:XX] INFO: ============================================================
[2026-01-18 XX:XX:XX] INFO: 🚀 All components initialized successfully
[2026-01-18 XX:XX:XX] INFO: Starting TradingOrchestrator...
[2026-01-18 XX:XX:XX] INFO: ============================================================
[2026-01-18 XX:XX:XX] INFO: ✓ TradingOrchestrator started
[2026-01-18 XX:XX:XX] INFO:
Listening for signals... (Press Ctrl+C to stop)
```

Verify:
- No errors during startup
- Telegram receives startup message
- Alpaca connection succeeds
- All components initialize

Press Ctrl+C and verify graceful shutdown:
```
[2026-01-18 XX:XX:XX] INFO:
⚠️  Shutdown requested...
[2026-01-18 XX:XX:XX] INFO: Shutting down gracefully...
[2026-01-18 XX:XX:XX] INFO: ✓ Orchestrator stopped
[2026-01-18 XX:XX:XX] INFO: ✓ Alpaca disconnected
[2026-01-18 XX:XX:XX] INFO: ✓ Shutdown notification sent
[2026-01-18 XX:XX:XX] INFO: ============================================================
[2026-01-18 XX:XX:XX] INFO: ✓ Shutdown complete
[2026-01-18 XX:XX:XX] INFO: ============================================================
```

**Step 4: Commit integration test**

```bash
git add tests/integration/test_main_startup.py
git commit -m "test: add integration test for main.py startup

- Test complete 6-phase initialization sequence
- Mock all external dependencies
- Verify graceful shutdown sequence
- Manual verification passed

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Update Documentation

**Files:**
- Modify: `docs/SIGUIENTES-PASOS.md`

**Step 1: Update next steps document**

Update section "## 2. Actualizar main.py con Pipeline Completo" in `docs/SIGUIENTES-PASOS.md`:

```markdown
## 2. Actualizar main.py con Pipeline Completo

✅ **COMPLETADO** - El `main.py` ahora integra todos los componentes del sistema.

### Cambios Implementados

1. **Inicialización Secuencial en 6 Fases:**
   - Fase 1: Configuración (env vars, settings, data dirs)
   - Fase 2: Infraestructura (Alpaca, Telegram)
   - Fase 3: Análisis (FinTwitBERT, Claude)
   - Fase 4: Collectors (Stocktwits, Reddit opcional)
   - Fase 5: Pipeline (Scorer, Validator, Gate, Risk, Journal, Executor)
   - Fase 6: Orchestrator (coordinador principal)

2. **Funciones Helper:**
   - `validate_env_vars()` - Valida variables requeridas
   - `create_data_dirs()` - Crea estructura de directorios
   - `print_startup_banner()` - Banner de inicio

3. **Manejo de Errores:**
   - Fail-fast: El sistema se detiene inmediatamente si falla un componente crítico
   - Mensajes claros indicando qué falló y cómo arreglarlo
   - Reddit opcional: Se salta si no hay credenciales (warning, no error)

4. **Shutdown Graceful:**
   - Detiene orchestrator procesando mensajes pendientes
   - Desconecta Alpaca
   - Envía notificación final a Telegram
   - Logs claros de cada paso

5. **Observabilidad:**
   - Logging estructurado con timestamps
   - Checkmark ✓ para cada componente inicializado
   - Mensajes informativos durante operación

### Cómo Ejecutar

```bash
# Crear directorios de datos (primera vez)
mkdir -p data/trades data/signals data/cache data/backtest_results

# Ejecutar sistema
uv run python main.py
```

### Salida Esperada

Ver `docs/plans/2026-01-18-main-integration-design.md` sección "Execution" para la salida completa esperada.
```

**Step 2: Commit documentation**

```bash
git add docs/SIGUIENTES-PASOS.md
git commit -m "docs: update SIGUIENTES-PASOS.md - main.py integration complete

Mark Section 2 as completed with implementation details

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Final Cleanup & Test Suite

**Files:**
- Run all tests

**Step 1: Run complete test suite**

```bash
uv run pytest -v
```

Expected: 664+ tests passing (all original + new tests)

**Step 2: Run with coverage**

```bash
uv run pytest --cov=src --cov=main --cov-report=term-missing
```

Verify main.py has reasonable coverage (>80%)

**Step 3: Lint check**

```bash
uv run ruff check src/ main.py
```

Expected: No errors

**Step 4: Format code**

```bash
uv run black src/ main.py tests/
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: format code and verify tests

- All 664+ tests passing
- Code formatted with black
- Lint checks passed
- Main.py integration complete

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Success Criteria

✅ All 6 initialization phases implemented
✅ Helper functions tested
✅ Fail-fast error handling throughout
✅ Graceful shutdown sequence
✅ Full logging and observability
✅ Integration test passes
✅ Manual dry run successful
✅ All 664+ tests passing
✅ Documentation updated
✅ Code formatted and linted

---

## Next Steps After Implementation

1. **Paper Trading Test (1-2 weeks)**
   - Run system during market hours
   - Monitor for errors and edge cases
   - Verify journal entries created
   - Check Telegram notifications

2. **Threshold Tuning**
   - Adjust `settings.yaml` based on paper trading results
   - Review scoring thresholds
   - Tune circuit breaker limits

3. **Production Readiness**
   - Security audit
   - Performance testing
   - Disaster recovery procedures
   - Monitoring setup

---

*Plan created: 2026-01-18*
*Estimated time: 2-3 hours*
