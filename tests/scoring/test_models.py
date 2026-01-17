# tests/scoring/test_models.py
"""Tests for scoring system data models."""
import pytest
from datetime import datetime, timezone

from src.scoring.models import (
    Direction,
    ScoreTier,
    ScoreComponents,
    TradeRecommendation,
)


class TestDirectionEnum:
    """Tests for Direction enum."""

    def test_direction_enum_values(self):
        """Test that Direction enum has correct string values."""
        assert Direction.LONG.value == "long"
        assert Direction.SHORT.value == "short"
        assert Direction.NEUTRAL.value == "neutral"

    def test_direction_enum_from_string(self):
        """Test that Direction can be created from string values."""
        assert Direction("long") == Direction.LONG
        assert Direction("short") == Direction.SHORT
        assert Direction("neutral") == Direction.NEUTRAL


class TestScoreTierEnum:
    """Tests for ScoreTier enum."""

    def test_score_tier_enum_values(self):
        """Test that ScoreTier enum has correct string values."""
        assert ScoreTier.STRONG.value == "strong"
        assert ScoreTier.MODERATE.value == "moderate"
        assert ScoreTier.WEAK.value == "weak"
        assert ScoreTier.NO_TRADE.value == "no_trade"

    def test_score_tier_from_score_strong(self):
        """Test from_score returns STRONG for score >= 80."""
        assert ScoreTier.from_score(80.0) == ScoreTier.STRONG
        assert ScoreTier.from_score(90.0) == ScoreTier.STRONG
        assert ScoreTier.from_score(100.0) == ScoreTier.STRONG

    def test_score_tier_from_score_moderate(self):
        """Test from_score returns MODERATE for 60 <= score < 80."""
        assert ScoreTier.from_score(60.0) == ScoreTier.MODERATE
        assert ScoreTier.from_score(70.0) == ScoreTier.MODERATE
        assert ScoreTier.from_score(79.9) == ScoreTier.MODERATE

    def test_score_tier_from_score_weak(self):
        """Test from_score returns WEAK for 40 <= score < 60."""
        assert ScoreTier.from_score(40.0) == ScoreTier.WEAK
        assert ScoreTier.from_score(50.0) == ScoreTier.WEAK
        assert ScoreTier.from_score(59.9) == ScoreTier.WEAK

    def test_score_tier_from_score_no_trade(self):
        """Test from_score returns NO_TRADE for score < 40."""
        assert ScoreTier.from_score(0.0) == ScoreTier.NO_TRADE
        assert ScoreTier.from_score(20.0) == ScoreTier.NO_TRADE
        assert ScoreTier.from_score(39.9) == ScoreTier.NO_TRADE


class TestScoreComponents:
    """Tests for ScoreComponents dataclass."""

    def test_score_components_creation(self):
        """Test that ScoreComponents can be created with valid values."""
        components = ScoreComponents(
            sentiment_score=75.0,
            technical_score=80.0,
            sentiment_weight=0.4,
            technical_weight=0.6,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=0.9,
        )
        assert components.sentiment_score == 75.0
        assert components.technical_score == 80.0
        assert components.sentiment_weight == 0.4
        assert components.technical_weight == 0.6
        assert components.confluence_bonus == 0.1
        assert components.credibility_multiplier == 1.0
        assert components.time_factor == 0.9

    def test_score_components_boundary_values(self):
        """Test ScoreComponents with boundary values."""
        # Minimum values
        components_min = ScoreComponents(
            sentiment_score=0.0,
            technical_score=0.0,
            sentiment_weight=0.0,
            technical_weight=0.0,
            confluence_bonus=0.0,
            credibility_multiplier=0.8,
            time_factor=0.5,
        )
        assert components_min.sentiment_score == 0.0
        assert components_min.credibility_multiplier == 0.8
        assert components_min.time_factor == 0.5

        # Maximum values
        components_max = ScoreComponents(
            sentiment_score=100.0,
            technical_score=100.0,
            sentiment_weight=1.0,
            technical_weight=1.0,
            confluence_bonus=0.2,
            credibility_multiplier=1.2,
            time_factor=1.0,
        )
        assert components_max.sentiment_score == 100.0
        assert components_max.confluence_bonus == 0.2
        assert components_max.credibility_multiplier == 1.2


class TestTradeRecommendation:
    """Tests for TradeRecommendation dataclass."""

    def test_trade_recommendation_creation(self):
        """Test that TradeRecommendation can be created with all fields."""
        components = ScoreComponents(
            sentiment_score=75.0,
            technical_score=80.0,
            sentiment_weight=0.4,
            technical_weight=0.6,
            confluence_bonus=0.1,
            credibility_multiplier=1.0,
            time_factor=0.9,
        )
        timestamp = datetime.now(timezone.utc)

        recommendation = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=2.5,
            entry_price=175.50,
            stop_loss=172.00,
            take_profit=185.00,
            risk_reward_ratio=2.71,
            components=components,
            reasoning="Strong bullish sentiment with technical confirmation",
            timestamp=timestamp,
        )

        assert recommendation.symbol == "AAPL"
        assert recommendation.direction == Direction.LONG
        assert recommendation.score == 85.0
        assert recommendation.tier == ScoreTier.STRONG
        assert recommendation.position_size_percent == 2.5
        assert recommendation.entry_price == 175.50
        assert recommendation.stop_loss == 172.00
        assert recommendation.take_profit == 185.00
        assert recommendation.risk_reward_ratio == 2.71
        assert recommendation.components == components
        assert recommendation.reasoning == "Strong bullish sentiment with technical confirmation"
        assert recommendation.timestamp == timestamp

    def test_trade_recommendation_short_direction(self):
        """Test TradeRecommendation with SHORT direction."""
        components = ScoreComponents(
            sentiment_score=70.0,
            technical_score=65.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.05,
            credibility_multiplier=0.95,
            time_factor=0.8,
        )

        recommendation = TradeRecommendation(
            symbol="TSLA",
            direction=Direction.SHORT,
            score=65.0,
            tier=ScoreTier.MODERATE,
            position_size_percent=1.5,
            entry_price=250.00,
            stop_loss=260.00,
            take_profit=230.00,
            risk_reward_ratio=2.0,
            components=components,
            reasoning="Bearish technical pattern with negative sentiment",
            timestamp=datetime.now(timezone.utc),
        )

        assert recommendation.direction == Direction.SHORT
        assert recommendation.tier == ScoreTier.MODERATE

    def test_trade_recommendation_neutral_direction(self):
        """Test TradeRecommendation with NEUTRAL direction (no trade)."""
        components = ScoreComponents(
            sentiment_score=50.0,
            technical_score=50.0,
            sentiment_weight=0.5,
            technical_weight=0.5,
            confluence_bonus=0.0,
            credibility_multiplier=1.0,
            time_factor=1.0,
        )

        recommendation = TradeRecommendation(
            symbol="SPY",
            direction=Direction.NEUTRAL,
            score=35.0,
            tier=ScoreTier.NO_TRADE,
            position_size_percent=0.0,
            entry_price=450.00,
            stop_loss=0.0,
            take_profit=0.0,
            risk_reward_ratio=0.0,
            components=components,
            reasoning="Mixed signals, no clear direction",
            timestamp=datetime.now(timezone.utc),
        )

        assert recommendation.direction == Direction.NEUTRAL
        assert recommendation.tier == ScoreTier.NO_TRADE
        assert recommendation.position_size_percent == 0.0
