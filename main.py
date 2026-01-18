# main.py
"""Main entry point for the intraday trading system."""
import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.config.settings import Settings
from src.collectors import CollectorManager, StocktwitsCollector, RedditCollector
from src.execution.alpaca_client import AlpacaClient
from src.notifications import TelegramNotifier, AlertFormatter
from src.notifications.models import Alert, AlertType
from src.analyzers import AnalyzerManager, SentimentAnalyzer, ClaudeAnalyzer
from src.models.social_message import SocialMessage


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
        )
        logger.info("✓ FinTwitBERT model loaded")

    except Exception as e:
        logger.error(f"Failed to load FinTwitBERT model: {e}")
        logger.error("Check internet connection for model download")
        sys.exit(1)

    # Initialize ClaudeAnalyzer
    try:
        claude_analyzer = ClaudeAnalyzer(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=settings.analyzers.claude.model,
            max_tokens=settings.analyzers.claude.max_tokens,
            rate_limit_per_minute=settings.analyzers.claude.rate_limit_per_minute,
        )
        # Test connection with simple message
        test_msg = SocialMessage(
            source="twitter",
            source_id="test_001",
            author="test_user",
            content="Test message for API verification",
            timestamp=datetime.now(),
            url="https://test.com"
        )
        _ = claude_analyzer.analyze(test_msg)
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

    # Initialize collectors based on settings
    collectors = []

    if settings.collectors.twitter.enabled:
        twitter = TwitterCollector(
            accounts_to_follow=["unusual_whales", "FirstSquawk", "zerohedge"],
            refresh_interval=settings.collectors.twitter.refresh_interval_seconds,
        )
        collectors.append(twitter)
        print("Twitter collector enabled")

    if settings.collectors.reddit.enabled:
        reddit = RedditCollector(
            subreddits=["wallstreetbets", "stocks", "options"],
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
