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
