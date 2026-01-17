# tests/config/test_analyzer_settings.py
import pytest
from src.config.settings import Settings


class TestAnalyzerSettings:
    def test_settings_has_analyzer_config(self):
        settings = Settings()
        assert hasattr(settings, "analyzers")

    def test_sentiment_analyzer_defaults(self):
        settings = Settings()
        assert settings.analyzers.sentiment.model == "StephanAkkerman/FinTwitBERT-sentiment"
        assert settings.analyzers.sentiment.batch_size == 32
        assert settings.analyzers.sentiment.min_confidence == 0.7

    def test_claude_analyzer_defaults(self):
        settings = Settings()
        assert settings.analyzers.claude.enabled is True
        assert settings.analyzers.claude.model == "claude-sonnet-4-20250514"
        assert settings.analyzers.claude.max_tokens == 1000
        assert settings.analyzers.claude.rate_limit_per_minute == 20
