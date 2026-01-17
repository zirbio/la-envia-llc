# src/scoring/recommendation_builder.py
"""Recommendation builder for trade recommendations."""
from datetime import datetime

from .models import Direction, ScoreTier, ScoreComponents, TradeRecommendation


class RecommendationBuilder:
    """Builds trade recommendations with entry, stop, and target."""

    def __init__(
        self,
        default_stop_loss_percent: float = 2.0,
        default_risk_reward_ratio: float = 2.0,
        tier_strong_threshold: int = 80,
        tier_moderate_threshold: int = 60,
        tier_weak_threshold: int = 40,
        position_size_strong: float = 100.0,
        position_size_moderate: float = 50.0,
        position_size_weak: float = 25.0,
    ):
        """Initialize RecommendationBuilder with configurable thresholds.

        Args:
            default_stop_loss_percent: Default stop loss as percentage of entry price.
            default_risk_reward_ratio: Default risk/reward ratio for take profit.
            tier_strong_threshold: Minimum score for STRONG tier.
            tier_moderate_threshold: Minimum score for MODERATE tier.
            tier_weak_threshold: Minimum score for WEAK tier.
            position_size_strong: Position size percentage for STRONG tier.
            position_size_moderate: Position size percentage for MODERATE tier.
            position_size_weak: Position size percentage for WEAK tier.
        """
        self.default_stop_loss_percent = default_stop_loss_percent
        self.default_risk_reward_ratio = default_risk_reward_ratio
        self.tier_strong_threshold = tier_strong_threshold
        self.tier_moderate_threshold = tier_moderate_threshold
        self.tier_weak_threshold = tier_weak_threshold
        self.position_size_strong = position_size_strong
        self.position_size_moderate = position_size_moderate
        self.position_size_weak = position_size_weak

    def get_tier(self, score: float) -> ScoreTier:
        """Get score tier from numeric score.

        Args:
            score: Numeric score from 0-100.

        Returns:
            ScoreTier based on configured thresholds.
        """
        if score >= self.tier_strong_threshold:
            return ScoreTier.STRONG
        elif score >= self.tier_moderate_threshold:
            return ScoreTier.MODERATE
        elif score >= self.tier_weak_threshold:
            return ScoreTier.WEAK
        else:
            return ScoreTier.NO_TRADE

    def get_position_size(self, tier: ScoreTier) -> float:
        """Get position size percentage for tier.

        Args:
            tier: The score tier.

        Returns:
            Position size as percentage (0-100).
        """
        if tier == ScoreTier.STRONG:
            return self.position_size_strong
        elif tier == ScoreTier.MODERATE:
            return self.position_size_moderate
        elif tier == ScoreTier.WEAK:
            return self.position_size_weak
        else:
            return 0.0

    def calculate_levels(
        self,
        entry_price: float,
        direction: Direction,
        stop_loss_percent: float = None,
        risk_reward_ratio: float = None,
    ) -> tuple[float, float, float]:
        """Calculate entry, stop loss, and take profit.

        For LONG direction:
            - stop_loss = entry_price * (1 - stop_loss_percent/100)
            - take_profit = entry_price + (entry_price - stop_loss) * risk_reward_ratio

        For SHORT direction:
            - stop_loss = entry_price * (1 + stop_loss_percent/100)
            - take_profit = entry_price - (stop_loss - entry_price) * risk_reward_ratio

        Args:
            entry_price: The entry price for the trade.
            direction: Trade direction (LONG, SHORT, or NEUTRAL).
            stop_loss_percent: Stop loss as percentage. Defaults to configured value.
            risk_reward_ratio: Risk/reward ratio. Defaults to configured value.

        Returns:
            Tuple of (entry, stop_loss, take_profit).
        """
        if stop_loss_percent is None:
            stop_loss_percent = self.default_stop_loss_percent
        if risk_reward_ratio is None:
            risk_reward_ratio = self.default_risk_reward_ratio

        if direction == Direction.SHORT:
            # SHORT: stop loss above entry, take profit below
            stop_loss = entry_price * (1 + stop_loss_percent / 100)
            risk = stop_loss - entry_price
            take_profit = entry_price - risk * risk_reward_ratio
        else:
            # LONG or NEUTRAL: stop loss below entry, take profit above
            stop_loss = entry_price * (1 - stop_loss_percent / 100)
            risk = entry_price - stop_loss
            take_profit = entry_price + risk * risk_reward_ratio

        return (entry_price, stop_loss, take_profit)

    def build(
        self,
        symbol: str,
        direction: Direction,
        score: float,
        current_price: float,
        components: ScoreComponents,
        reasoning: str,
    ) -> TradeRecommendation:
        """Build complete trade recommendation.

        Args:
            symbol: Stock ticker symbol.
            direction: Trade direction.
            score: Overall score from 0-100.
            current_price: Current price of the asset.
            components: Score components breakdown.
            reasoning: Human-readable explanation.

        Returns:
            Complete TradeRecommendation with all fields populated.
        """
        tier = self.get_tier(score)
        position_size = self.get_position_size(tier)
        entry_price, stop_loss, take_profit = self.calculate_levels(
            entry_price=current_price,
            direction=direction,
        )

        return TradeRecommendation(
            symbol=symbol,
            direction=direction,
            score=score,
            tier=tier,
            position_size_percent=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=self.default_risk_reward_ratio,
            components=components,
            reasoning=reasoning,
            timestamp=datetime.now(),
        )
