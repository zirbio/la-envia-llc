"""Tests for main.py helper functions."""
import os
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import pytest


def test_validate_env_vars_success():
    """Test env var validation with all required vars present."""
    from main import validate_env_vars

    with patch.dict(os.environ, {
        "ALPACA_API_KEY": "test_key",
        "ALPACA_SECRET_KEY": "test_secret",
        "ANTHROPIC_API_KEY": "test_anthropic",
        "XAI_API_KEY": "test_xai",
    }):
        # Should not raise
        validate_env_vars()


def test_validate_env_vars_missing_alpaca_key():
    """Test env var validation fails when ALPACA_API_KEY missing."""
    from main import validate_env_vars

    with patch.dict(os.environ, {
        "ALPACA_SECRET_KEY": "test_secret",
        "ANTHROPIC_API_KEY": "test_anthropic",
        "XAI_API_KEY": "test_xai",
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
        "XAI_API_KEY=test\n"
    )

    # Create temporary settings.yaml
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("system:\n  name: Test\n  mode: paper\n")

    with patch.dict(os.environ, {
        "ALPACA_API_KEY": "test",
        "ALPACA_SECRET_KEY": "test",
        "ANTHROPIC_API_KEY": "test",
        "XAI_API_KEY": "test",
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
async def test_initialize_claude_analyzer_success():
    """Test successful Claude analyzer initialization."""
    from main import initialize_claude_analyzer
    from src.config.settings import Settings

    settings = Settings.from_yaml(Path("config/settings.yaml"))

    with patch("main.ClaudeAnalyzer") as mock_claude:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            # Mock successful init
            mock_claude_instance = mock_claude.return_value
            mock_claude_instance.analyze.return_value = None

            claude_analyzer = await initialize_claude_analyzer(settings)

            assert claude_analyzer is not None
            mock_claude_instance.analyze.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_grok_collector_success():
    """Test successful Grok collector initialization."""
    from main import initialize_grok_collector
    from src.config.settings import Settings

    settings = Settings.from_yaml(Path("config/settings.yaml"))

    with patch("main.GrokCollector") as mock_grok:
        with patch.dict(os.environ, {"XAI_API_KEY": "test_xai_key"}):
            # Mock successful init and connect
            mock_grok_instance = mock_grok.return_value
            mock_grok_instance.connect = AsyncMock()

            grok_collector = await initialize_grok_collector(settings)

            assert grok_collector is not None
            mock_grok_instance.connect.assert_called_once()


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
    assert "profile_store" in components
    assert "credibility_manager" in components
    assert "outcome_tracker" in components


def test_initialize_orchestrator_success():
    """Test successful orchestrator initialization."""
    from main import initialize_orchestrator
    from src.config.settings import Settings

    settings = Settings.from_yaml(Path("config/settings.yaml"))

    # Create mocks for all dependencies
    mock_grok_collector = MagicMock()
    mock_claude_analyzer = MagicMock()
    mock_components = {
        "scorer": MagicMock(),
        "validator": MagicMock(),
        "gate": MagicMock(),
        "risk_manager": MagicMock(),
        "journal": MagicMock(),
        "executor": MagicMock(),
        "credibility_manager": MagicMock(),
        "outcome_tracker": MagicMock(),
    }

    with patch("main.TradingOrchestrator") as mock_orch:
        orchestrator = initialize_orchestrator(
            settings,
            mock_grok_collector,
            mock_claude_analyzer,
            mock_components,
        )

        assert orchestrator is not None
        mock_orch.assert_called_once()


@pytest.mark.asyncio
async def test_main_full_startup():
    """Test full main() startup sequence."""
    from main import main

    with patch("main.load_and_validate_config") as mock_config, \
         patch("main.print_startup_banner"), \
         patch("main.initialize_infrastructure") as mock_infra, \
         patch("main.initialize_claude_analyzer") as mock_claude, \
         patch("main.initialize_grok_collector") as mock_grok, \
         patch("main.initialize_pipeline_components") as mock_pipeline, \
         patch("main.initialize_orchestrator") as mock_orchestrator:

        # Setup mocks
        mock_settings = MagicMock()
        mock_config.return_value = mock_settings

        mock_alpaca = MagicMock()
        mock_alpaca.disconnect = AsyncMock()
        mock_telegram = MagicMock()
        mock_telegram.send_alert = AsyncMock()
        mock_infra.return_value = (mock_alpaca, mock_telegram)

        mock_claude_analyzer = MagicMock()
        mock_claude.return_value = mock_claude_analyzer

        mock_grok_collector = MagicMock()
        mock_grok_collector.disconnect = AsyncMock()
        mock_grok.return_value = mock_grok_collector

        mock_components = {
            "scorer": MagicMock(),
            "validator": MagicMock(),
            "gate": MagicMock(),
            "risk_manager": MagicMock(),
            "journal": MagicMock(),
            "executor": MagicMock(),
            "credibility_manager": MagicMock(),
            "outcome_tracker": MagicMock(),
        }
        mock_pipeline.return_value = mock_components

        mock_orch = MagicMock()
        mock_orch.start = AsyncMock()
        mock_orch.stop = AsyncMock()
        mock_orchestrator.return_value = mock_orch

        # Mock KeyboardInterrupt to exit gracefully
        mock_orch.start.side_effect = KeyboardInterrupt()

        await main()

        # Verify all phases called
        mock_config.assert_called_once()
        mock_infra.assert_called_once()
        mock_claude.assert_called_once()
        mock_grok.assert_called_once()
        mock_pipeline.assert_called_once()
        mock_orchestrator.assert_called_once()
        mock_orch.start.assert_called_once()
        mock_orch.stop.assert_called_once()
