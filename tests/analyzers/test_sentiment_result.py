# tests/analyzers/test_sentiment_result.py
import pytest
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel


class TestSentimentResult:
    def test_create_bullish_result(self):
        result = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.92,
            confidence=0.88,
        )
        assert result.label == SentimentLabel.BULLISH
        assert result.score == 0.92
        assert result.confidence == 0.88

    def test_create_bearish_result(self):
        result = SentimentResult(
            label=SentimentLabel.BEARISH,
            score=0.15,
            confidence=0.95,
        )
        assert result.label == SentimentLabel.BEARISH
        assert result.is_confident(min_confidence=0.7)

    def test_is_confident_above_threshold(self):
        result = SentimentResult(
            label=SentimentLabel.BULLISH,
            score=0.85,
            confidence=0.80,
        )
        assert result.is_confident(min_confidence=0.7) is True
        assert result.is_confident(min_confidence=0.85) is False

    def test_neutral_label(self):
        result = SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.50,
            confidence=0.60,
        )
        assert result.label == SentimentLabel.NEUTRAL
