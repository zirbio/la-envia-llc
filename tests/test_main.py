"""Tests for main.py helper functions."""
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock
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
        with pytest.raises(SystemExit):
            validate_env_vars()


def test_create_data_dirs(tmp_path, monkeypatch):
    """Test data directory creation."""
    from main import create_data_dirs

    # Change to temp directory to avoid creating dirs in actual project
    monkeypatch.chdir(tmp_path)
    create_data_dirs()

    assert (tmp_path / "data" / "trades").exists()
    assert (tmp_path / "data" / "signals").exists()
    assert (tmp_path / "data" / "cache").exists()
    assert (tmp_path / "data" / "backtest_results").exists()


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

    with patch.dict(os.environ, {
        "ALPACA_API_KEY": "test",
        "ALPACA_SECRET_KEY": "test",
        "ANTHROPIC_API_KEY": "test",
    }):
        with patch("main.load_dotenv"):
            with patch("main.create_data_dirs"):
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


@pytest.mark.asyncio
async def test_initialize_analyzers_success():
    """Test successful analyzer initialization."""
    from main import initialize_analyzers
    from src.config.settings import Settings

    settings = Settings.from_yaml(Path("config/settings.yaml"))

    with patch("main.SentimentAnalyzer") as mock_sent:
        with patch("main.ClaudeAnalyzer") as mock_claude:
            with patch("main.AnalyzerManager") as mock_manager:
                with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
                    # Mock successful init
                    mock_sent_instance = mock_sent.return_value
                    mock_claude_instance = mock_claude.return_value
                    mock_claude_instance.analyze.return_value = None

                    analyzer_manager = await initialize_analyzers(settings)

                    assert analyzer_manager is not None
                    mock_claude_instance.analyze.assert_called_once()
