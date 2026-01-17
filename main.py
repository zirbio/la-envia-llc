# main.py
"""Main entry point for the intraday trading system."""
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
