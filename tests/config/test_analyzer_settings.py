# tests/config/test_analyzer_settings.py
import pytest
from pydantic import ValidationError
from src.config.settings import (
    Settings,
    SentimentAnalyzerSettings,
    ClaudeAnalyzerSettings,
)


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

    def test_claude_use_for_defaults(self):
        settings = Settings()
        assert "catalyst_classification" in settings.analyzers.claude.use_for
        assert "risk_assessment" in settings.analyzers.claude.use_for
        assert "context_analysis" in settings.analyzers.claude.use_for
        assert len(settings.analyzers.claude.use_for) == 3

    def test_sentiment_batch_size_validation_negative(self):
        with pytest.raises(ValidationError):
            SentimentAnalyzerSettings(batch_size=0)

    def test_sentiment_batch_size_validation_too_large(self):
        with pytest.raises(ValidationError):
            SentimentAnalyzerSettings(batch_size=300)

    def test_sentiment_min_confidence_validation_negative(self):
        with pytest.raises(ValidationError):
            SentimentAnalyzerSettings(min_confidence=-0.1)

    def test_sentiment_min_confidence_validation_above_one(self):
        with pytest.raises(ValidationError):
            SentimentAnalyzerSettings(min_confidence=1.5)

    def test_claude_max_tokens_validation_too_small(self):
        with pytest.raises(ValidationError):
            ClaudeAnalyzerSettings(max_tokens=50)

    def test_claude_rate_limit_validation_zero(self):
        with pytest.raises(ValidationError):
            ClaudeAnalyzerSettings(rate_limit_per_minute=0)
