# src/analyzers/analyzed_message.py
from typing import Optional

from pydantic import BaseModel

from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_result import ClaudeAnalysisResult
from src.models.social_message import SocialMessage


class AnalyzedMessage(BaseModel):
    """A social message with sentiment and optional deep analysis."""

    message: SocialMessage
    sentiment: SentimentResult
    deep_analysis: Optional[ClaudeAnalysisResult] = None

    def _has_actionable_sentiment(self, min_confidence: float) -> bool:
        """Check if sentiment is confident and non-neutral."""
        return (
            self.sentiment.is_confident(min_confidence)
            and self.sentiment.label != SentimentLabel.NEUTRAL
        )

    def is_high_signal(self, min_sentiment_confidence: float = 0.7) -> bool:
        """Check if this message is a high-signal opportunity."""
        return self._has_actionable_sentiment(min_sentiment_confidence)

    def requires_deep_analysis(self, min_sentiment_confidence: float = 0.7) -> bool:
        """Check if this message should get deep analysis."""
        return (
            self.deep_analysis is None
            and self._has_actionable_sentiment(min_sentiment_confidence)
        )

    def get_tickers(self, exclude_crypto: bool = True) -> list[str]:
        """Get tickers from the underlying message."""
        return self.message.extract_tickers(exclude_crypto=exclude_crypto)
