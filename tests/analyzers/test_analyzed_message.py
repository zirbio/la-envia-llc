# tests/analyzers/test_analyzed_message.py
import pytest
from datetime import datetime, timezone
from src.analyzers.analyzed_message import AnalyzedMessage
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult, CatalystType, RiskLevel
from src.models.social_message import SocialMessage, SourceType


class TestAnalyzedMessage:
    @pytest.fixture
    def sample_message(self):
        return SocialMessage(
            source=SourceType.TWITTER,
            source_id="123",
            author="unusual_whales",
            content="Large $NVDA call sweep",
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def sample_sentiment(self):
        return SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.92,
            confidence=0.88,
        )

    def test_create_with_sentiment_only(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.message == sample_message
        assert analyzed.sentiment == sample_sentiment
        assert analyzed.deep_analysis is None

    def test_create_with_deep_analysis(self, sample_message, sample_sentiment):
        deep = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            context_summary="test",
            recommendation="valid",
        )
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
            deep_analysis=deep,
        )
        assert analyzed.deep_analysis == deep

    def test_is_high_signal_true(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.is_high_signal(min_sentiment_confidence=0.7)

    def test_is_high_signal_false_neutral_sentiment(self, sample_message):
        """is_high_signal returns False for NEUTRAL sentiment."""
        neutral_sentiment = SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.50,
            confidence=0.95,
        )
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=neutral_sentiment,
        )
        assert not analyzed.is_high_signal(min_sentiment_confidence=0.7)

    def test_is_high_signal_false_low_confidence(self, sample_message):
        """is_high_signal returns False for low confidence."""
        low_confidence_sentiment = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.85,
            confidence=0.5,
        )
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=low_confidence_sentiment,
        )
        assert not analyzed.is_high_signal(min_sentiment_confidence=0.7)

    def test_requires_deep_analysis_true(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.requires_deep_analysis(min_sentiment_confidence=0.7)

    def test_requires_deep_analysis_false_already_analyzed(
        self, sample_message, sample_sentiment
    ):
        """requires_deep_analysis returns False when deep_analysis is present."""
        deep = ClaudeAnalysisResult(
            catalyst_type=CatalystType.INSTITUTIONAL_FLOW,
            catalyst_confidence=0.85,
            risk_level=RiskLevel.LOW,
            context_summary="test",
            recommendation="valid",
        )
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
            deep_analysis=deep,
        )
        assert not analyzed.requires_deep_analysis(min_sentiment_confidence=0.7)

    def test_get_tickers(self, sample_message, sample_sentiment):
        analyzed = AnalyzedMessage(
            message=sample_message,
            sentiment=sample_sentiment,
        )
        assert analyzed.get_tickers() == ["NVDA"]

    def test_get_tickers_include_crypto(self, sample_sentiment):
        """get_tickers includes crypto when exclude_crypto=False."""
        message_with_crypto = SocialMessage(
            source=SourceType.TWITTER,
            source_id="456",
            author="crypto_whale",
            content="Loading up on $BTC and $NVDA today",
            timestamp=datetime.now(timezone.utc),
        )
        analyzed = AnalyzedMessage(
            message=message_with_crypto,
            sentiment=sample_sentiment,
        )
        # With exclude_crypto=True (default), BTC should be excluded
        assert analyzed.get_tickers(exclude_crypto=True) == ["NVDA"]
        # With exclude_crypto=False, BTC should be included
        assert analyzed.get_tickers(exclude_crypto=False) == ["BTC", "NVDA"]
