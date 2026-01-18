"""Integration test for main.py startup sequence."""
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import os
from src.config.settings import Settings


@pytest.mark.asyncio
async def test_full_startup_sequence():
    """Test complete startup sequence with all phases."""
    # Load actual settings to avoid complex mocking
    settings = Settings.from_yaml(Path("config/settings.yaml"))

    # Mock all external dependencies
    with patch("main.load_dotenv"):
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": "test_key",
            "ALPACA_SECRET_KEY": "test_secret",
            "ANTHROPIC_API_KEY": "test_anthropic",
            "TELEGRAM_BOT_TOKEN": "test_token",
            "TELEGRAM_CHAT_ID": "123456",
        }):
            with patch("main.Settings.from_yaml", return_value=settings):
                with patch("main.AlpacaClient") as mock_alpaca:
                    with patch("main.TelegramNotifier") as mock_telegram:
                        with patch("main.SentimentAnalyzer"):
                            with patch("main.ClaudeAnalyzer") as mock_claude:
                                with patch("main.TradingOrchestrator") as mock_orch:
                                    # Setup mocks
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
                                    claude_instance.analyze = MagicMock(
                                        return_value=MagicMock()
                                    )

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
                                    claude_instance.analyze.assert_called_once()
                                    orch_instance.start.assert_called_once()
                                    orch_instance.stop.assert_called_once()
                                    alpaca_instance.disconnect.assert_called_once()
