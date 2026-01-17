# src/analyzers/sentiment_result.py
from enum import Enum

from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SentimentResult(BaseModel):
    """Result from sentiment analysis."""

    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0, description="Sentiment score 0-1, higher=more bullish")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in prediction")

    def is_confident(self, min_confidence: float = 0.7) -> bool:
        """Check if result meets minimum confidence threshold."""
        return self.confidence >= min_confidence
