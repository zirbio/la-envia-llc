# main.py
"""Main entry point for the intraday trading system."""
import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add src directory to Python path for internal imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

from src.config.settings import Settings
from src.collectors import GrokCollector
from src.execution.alpaca_client import AlpacaClient
from src.notifications import TelegramNotifier, AlertFormatter
from src.notifications.models import Alert, AlertType
from src.analyzers import ClaudeAnalyzer
from src.models.social_message import SocialMessage
from src.validators import TechnicalValidator
from src.scoring import (
    SignalScorer,
    SourceProfileStore,
    DynamicCredibilityManager,
    SignalOutcomeTracker,
    TimeFactorCalculator,
    ConfluenceDetector,
    DynamicWeightCalculator,
    RecommendationBuilder,
)
from src.gate import MarketGate
from src.risk import RiskManager
from src.journal import JournalManager
from src.execution import TradeExecutor
from src.orchestrator import TradingOrchestrator


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
        "XAI_API_KEY",
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
        Path("data/sources"),
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Data directories verified")


def print_startup_banner(settings: Settings) -> None:
    """Print system startup banner."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.system.name}")
    logger.info(f"Mode: {settings.system.mode}")
    logger.info(f"Version: {settings.system.version}")
    logger.info("=" * 60)


def load_and_validate_config() -> Settings:
    """Load and validate configuration.

    Returns:
        Settings object loaded from YAML.

    Raises:
        SystemExit: If config file missing, env vars invalid, or YAML parsing fails.
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
    try:
        settings = Settings.from_yaml(config_path)
        logger.info("✓ Settings loaded from config/settings.yaml")
    except Exception as e:
        logger.error(f"Failed to parse settings.yaml: {e}")
        sys.exit(1)

    # Create data directories
    create_data_dirs()

    return settings


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
        await telegram.send_alert(
            Alert(
                alert_type=AlertType.SYSTEM,
                message="🚀 System starting up..."
            )
        )
        logger.info("✓ Telegram connected (sent startup message)")

    except Exception as e:
        logger.error(f"Failed to connect to Telegram: {e}")
        logger.error("Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        sys.exit(1)

    return alpaca_client, telegram


async def initialize_claude_analyzer(settings: Settings) -> ClaudeAnalyzer:
    """Initialize Claude analyzer for deep analysis.

    Note: Grok provides sentiment analysis, so FinTwitBERT is no longer needed.

    Args:
        settings: Loaded settings object.

    Returns:
        ClaudeAnalyzer instance.

    Raises:
        SystemExit: If API connection fails.
    """
    try:
        claude_analyzer = ClaudeAnalyzer(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=settings.analyzers.claude.model,
            max_tokens=settings.analyzers.claude.max_tokens,
            rate_limit_per_minute=settings.analyzers.claude.rate_limit_per_minute,
        )
        # Test connection with realistic trading message
        test_msg = SocialMessage(
            source="twitter",
            source_id="test_001",
            author="unusual_whales",
            content="$AAPL breaking out above resistance with massive volume spike. Bullish setup.",
            timestamp=datetime.now(),
            url="https://x.com/unusual_whales/status/test_001"
        )
        _ = claude_analyzer.analyze(test_msg)
        logger.info("✓ Claude API verified")

    except Exception as e:
        logger.error(f"Claude API test failed: {e}")
        logger.error("Check ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    return claude_analyzer


async def initialize_grok_collector(settings: Settings) -> GrokCollector:
    """Initialize and connect Grok collector for X/Twitter data.

    Args:
        settings: Loaded settings object.

    Returns:
        Connected GrokCollector instance.

    Raises:
        SystemExit: If initialization or connection fails.
    """
    try:
        grok_settings = settings.collectors.grok
        grok = GrokCollector(
            api_key=os.getenv("XAI_API_KEY"),
            search_queries=grok_settings.search_queries,
            refresh_interval=grok_settings.refresh_interval_seconds,
            max_results_per_query=grok_settings.max_results_per_query,
        )
        await grok.connect()
        logger.info("✓ GrokCollector initialized and connected")
        return grok

    except Exception as e:
        logger.error(f"Failed to initialize GrokCollector: {e}")
        logger.error("Check XAI_API_KEY in .env")
        sys.exit(1)


def initialize_pipeline_components(
    settings: Settings,
    alpaca_client: AlpacaClient,
) -> dict:
    """Initialize pipeline components (Layers 3-6).

    Args:
        settings: Loaded settings object.
        alpaca_client: Connected Alpaca client.

    Returns:
        Dict with: scorer, validator, gate, risk_manager, journal, executor,
                   profile_store, credibility_manager, outcome_tracker.
    """
    # Dynamic credibility system (replaces static SourceCredibilityManager)
    profile_store = SourceProfileStore(data_dir=Path("data/sources"))
    logger.info("✓ SourceProfileStore initialized")

    credibility_manager = DynamicCredibilityManager(
        profile_store=profile_store,
        min_signals_for_ranking=settings.scoring.min_signals_for_dynamic,
        tier1_sources=settings.scoring.tier1_seeds,
        tier1_multiplier=settings.scoring.credibility_tier1_multiplier,
    )
    logger.info("✓ DynamicCredibilityManager initialized")

    outcome_tracker = SignalOutcomeTracker(
        credibility_manager=credibility_manager,
        alpaca_client=alpaca_client,
        evaluation_window_minutes=settings.scoring.evaluation_window_minutes,
        success_threshold_percent=settings.scoring.success_threshold_percent,
    )
    logger.info("✓ SignalOutcomeTracker initialized")

    # Layer 4: SignalScorer (requires sub-components)
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
        "profile_store": profile_store,
        "credibility_manager": credibility_manager,
        "outcome_tracker": outcome_tracker,
    }


def initialize_orchestrator(
    settings: Settings,
    grok_collector: GrokCollector,
    claude_analyzer: ClaudeAnalyzer,
    components: dict,
) -> TradingOrchestrator:
    """Initialize TradingOrchestrator.

    Args:
        settings: Loaded settings object.
        grok_collector: Initialized Grok collector.
        claude_analyzer: Initialized Claude analyzer.
        components: Dict with pipeline components.

    Returns:
        TradingOrchestrator instance.
    """
    orchestrator = TradingOrchestrator(
        grok_collector=grok_collector,
        claude_analyzer=claude_analyzer,
        technical_validator=components["validator"],
        signal_scorer=components["scorer"],
        risk_manager=components["risk_manager"],
        market_gate=components["gate"],
        trade_executor=components["executor"],
        credibility_manager=components["credibility_manager"],
        outcome_tracker=components["outcome_tracker"],
        settings=settings.orchestrator,
    )
    logger.info("✓ TradingOrchestrator initialized")

    return orchestrator


async def main():
    """Main entry point for the trading system."""
    # Phase 1: Configuration
    settings = load_and_validate_config()
    print_startup_banner(settings)

    # Phase 2: Core Infrastructure
    alpaca_client, telegram = await initialize_infrastructure(settings)

    # Phase 3: Claude Analyzer (Grok provides sentiment)
    claude_analyzer = await initialize_claude_analyzer(settings)

    # Phase 4: Grok Collector (replaces old collectors)
    grok_collector = await initialize_grok_collector(settings)

    # Phase 5: Pipeline Components (including dynamic credibility)
    components = initialize_pipeline_components(settings, alpaca_client)

    # Phase 6: Orchestrator
    orchestrator = initialize_orchestrator(
        settings,
        grok_collector,
        claude_analyzer,
        components,
    )

    # Get outcome tracker for evaluation loop
    outcome_tracker = components["outcome_tracker"]

    # Start system
    logger.info("=" * 60)
    logger.info("All components initialized successfully")
    logger.info("Starting TradingOrchestrator...")
    logger.info("=" * 60)

    try:
        await orchestrator.start()
        logger.info("✓ TradingOrchestrator started")
        logger.info("\nListening for signals... (Press Ctrl+C to stop)")

        # Keep running until interrupted
        while orchestrator.is_running:
            # Evaluate pending signal outcomes periodically
            await outcome_tracker.evaluate_pending()
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

    finally:
        # Graceful shutdown
        logger.info("Shutting down gracefully...")

        # Stop orchestrator (processes remaining messages)
        await orchestrator.stop()
        logger.info("✓ Orchestrator stopped")

        # Disconnect Grok collector
        await grok_collector.disconnect()
        logger.info("✓ GrokCollector disconnected")

        # Disconnect infrastructure
        await alpaca_client.disconnect()
        logger.info("✓ Alpaca disconnected")

        # Final notification
        try:
            await telegram.send_alert(
                Alert(
                    alert_type=AlertType.SYSTEM,
                    message="System shutdown complete"
                )
            )
            logger.info("✓ Shutdown notification sent")
        except Exception:
            pass  # Ignore telegram errors during shutdown

        logger.info("=" * 60)
        logger.info("✓ Shutdown complete")
        logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
