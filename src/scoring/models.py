# src/scoring/models.py
"""Data models for the scoring system."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    """Trade direction enumeration."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class ScoreTier(str, Enum):
    """Score tier classification for trade strength."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NO_TRADE = "no_trade"

    @classmethod
    def from_score(cls, score: float) -> "ScoreTier":
        """Get the appropriate tier for a given numeric score.

        Args:
            score: Numeric score from 0-100.

        Returns:
            ScoreTier based on thresholds:
                - score >= 80 -> STRONG
                - score >= 60 -> MODERATE
                - score >= 40 -> WEAK
                - score < 40 -> NO_TRADE
        """
        if score >= 80:
            return cls.STRONG
        elif score >= 60:
            return cls.MODERATE
        elif score >= 40:
            return cls.WEAK
        else:
            return cls.NO_TRADE


@dataclass
class ScoreComponents:
    """Components that make up a trade score.

    Attributes:
        sentiment_score: Sentiment analysis score (0-100).
        technical_score: Technical analysis score (0-100).
        sentiment_weight: Weight applied to sentiment score (0-1).
        technical_weight: Weight applied to technical score (0-1).
        confluence_bonus: Bonus for sentiment/technical alignment (0-0.2).
        credibility_multiplier: Source credibility adjustment (0.8-1.2).
        time_factor: Time decay factor for signal freshness (0.5-1.0).
    """

    sentiment_score: float  # 0-100
    technical_score: float  # 0-100
    sentiment_weight: float  # 0-1
    technical_weight: float  # 0-1
    confluence_bonus: float  # 0-0.2
    credibility_multiplier: float  # 0.8-1.2
    time_factor: float  # 0.5-1.0


@dataclass
class TradeRecommendation:
    """A complete trade recommendation with scoring details.

    Attributes:
        symbol: Stock ticker symbol.
        direction: Trade direction (LONG, SHORT, NEUTRAL).
        score: Overall score from 0-100.
        tier: Score tier classification.
        position_size_percent: Recommended position size as percentage (0-100).
        entry_price: Suggested entry price.
        stop_loss: Suggested stop loss price.
        take_profit: Suggested take profit price.
        risk_reward_ratio: Risk/reward ratio for the trade.
        components: Breakdown of score components.
        reasoning: Human-readable explanation of the recommendation.
        timestamp: When the recommendation was generated.
    """

    symbol: str
    direction: Direction
    score: float  # 0-100
    tier: ScoreTier
    position_size_percent: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    components: ScoreComponents
    reasoning: str
    timestamp: datetime
