# src/analyzers/analyzer_manager.py
from typing import Optional

from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.claude_analyzer import ClaudeAnalyzer
from src.analyzers.analyzed_message import AnalyzedMessage
from src.models.social_message import SocialMessage


class AnalyzerManager:
    """Orchestrates the analysis pipeline: sentiment -> deep analysis."""

    def __init__(
        self,
        sentiment_analyzer: SentimentAnalyzer,
        claude_analyzer: Optional[ClaudeAnalyzer] = None,
        min_sentiment_confidence: float = 0.7,
        enable_deep_analysis: bool = True,
    ):
        self.sentiment_analyzer = sentiment_analyzer
        self.claude_analyzer = claude_analyzer
        self.min_sentiment_confidence = min_sentiment_confidence
        self.enable_deep_analysis = enable_deep_analysis and claude_analyzer is not None

    def analyze(self, message: SocialMessage) -> AnalyzedMessage:
        """Analyze a single social message.

        Pipeline:
        1. Run sentiment analysis
        2. If high signal and deep analysis enabled, run Claude analysis
        """
        sentiment = self.sentiment_analyzer.analyze(message.content)

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
        """Analyze multiple messages in batch."""
        if not messages:
            return []

        contents = [m.content for m in messages]
        sentiments = self.sentiment_analyzer.analyze_batch(contents)

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
