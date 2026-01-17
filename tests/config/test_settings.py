# tests/config/test_settings.py
import pytest
import tempfile
import os
from pathlib import Path


class TestSettings:
    def test_load_settings_from_yaml(self, tmp_path):
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
        config_content = """
system:
  name: "Minimal"
"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content)

        from src.config.settings import Settings
        settings = Settings.from_yaml(config_file)

        assert settings.system.mode == "paper"
        assert settings.collectors.twitter.enabled is True

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
