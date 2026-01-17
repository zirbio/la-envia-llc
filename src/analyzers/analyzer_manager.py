# src/analyzers/analyzer_manager.py
import logging
from typing import Optional

from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.sentiment_result import SentimentResult, SentimentLabel
from src.analyzers.claude_analyzer import ClaudeAnalyzer
from src.analyzers.analyzed_message import AnalyzedMessage
from src.models.social_message import SocialMessage

logger = logging.getLogger(__name__)


class AnalyzerManager:
    """Orchestrates the analysis pipeline: sentiment -> deep analysis."""

    def __init__(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        claude_analyzer: Optional[ClaudeAnalyzer] = None,
        min_sentiment_confidence: float = 0.7,
        enable_deep_analysis: bool = True,
    ):
        """Initialize the AnalyzerManager.

        Args:
            sentiment_analyzer: Analyzer for sentiment classification.
            claude_analyzer: Optional analyzer for deep analysis.
            min_sentiment_confidence: Minimum confidence (0.0-1.0) to trigger deep analysis.
            enable_deep_analysis: Whether to perform deep analysis on high-signal messages.

        Raises:
            ValueError: If min_sentiment_confidence is not between 0.0 and 1.0.
        """
        if not 0.0 <= min_sentiment_confidence <= 1.0:
            raise ValueError(
                f"min_sentiment_confidence must be between 0.0 and 1.0, got {min_sentiment_confidence}"
            )

        self.sentiment_analyzer = sentiment_analyzer
        self.claude_analyzer = claude_analyzer
        self.min_sentiment_confidence = min_sentiment_confidence
        self.enable_deep_analysis = enable_deep_analysis and claude_analyzer is not None

    def _get_fallback_sentiment(self) -> SentimentResult:
        """Return neutral sentiment result for error cases."""
        return SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.5,
            confidence=0.0,
        )

    def analyze(self, message: SocialMessage) -> AnalyzedMessage:
        """Analyze a single social message.

        Pipeline:
        1. Run sentiment analysis
        2. If high signal and deep analysis enabled, run Claude analysis

        On sentiment analysis failure, returns neutral sentiment with zero confidence.
        """
        try:
            sentiment = self.sentiment_analyzer.analyze(message.content)
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for message {message.source_id}: {e}")
            sentiment = self._get_fallback_sentiment()

        analyzed = AnalyzedMessage(
            message=message,
            sentiment=sentiment,
        )

        if self.enable_deep_analysis and analyzed.requires_deep_analysis(
            self.min_sentiment_confidence
        ):
            deep_analysis = self.claude_analyzer.analyze(message)
            analyzed = AnalyzedMessage(
                message=message,
                sentiment=sentiment,
                deep_analysis=deep_analysis,
            )

        return analyzed

    def analyze_batch(self, messages: list[SocialMessage]) -> list[AnalyzedMessage]:
        """Analyze multiple messages in batch.

        Uses batch sentiment analysis for efficiency, then selectively
        applies deep analysis to high-signal messages.

        Note: Deep analysis is performed sequentially with rate limiting.
        This is by design to respect Claude API rate limits.

        On batch sentiment failure, falls back to individual analysis.
        """
        if not messages:
            return []

        # Batch sentiment analysis with fallback
        try:
            contents = [m.content for m in messages]
            sentiments = self.sentiment_analyzer.analyze_batch(contents)
        except Exception as e:
            logger.warning(f"Batch sentiment analysis failed: {e}. Falling back to individual analysis.")
            # Fall back to individual analysis
            return [self.analyze(m) for m in messages]

        results = []
        for message, sentiment in zip(messages, sentiments):
            analyzed = AnalyzedMessage(
                message=message,
                sentiment=sentiment,
            )

            if self.enable_deep_analysis and analyzed.requires_deep_analysis(
                self.min_sentiment_confidence
            ):
                deep_analysis = self.claude_analyzer.analyze(message)
                analyzed = AnalyzedMessage(
                    message=message,
                    sentiment=sentiment,
                    deep_analysis=deep_analysis,
                )

            results.append(analyzed)

        return results
