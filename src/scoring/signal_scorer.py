# src/scoring/signal_scorer.py
"""Signal scorer orchestrator for the scoring pipeline."""
from datetime import datetime

from src.analyzers.sentiment_result import SentimentLabel
from src.scoring.confluence_detector import ConfluenceDetector
from src.scoring.dynamic_weight_calculator import DynamicWeightCalculator
from src.scoring.models import Direction, ScoreComponents, ScoreTier, TradeRecommendation
from src.scoring.recommendation_builder import RecommendationBuilder
from src.scoring.source_credibility import SourceCredibilityManager
from src.scoring.time_factors import TimeFactorCalculator
from src.validators.models import ValidatedSignal, ValidationStatus


class SignalScorer:
    """Orchestrates the scoring pipeline.

    The SignalScorer combines all scoring components to produce a final
    trade recommendation from a validated signal.

    Pipeline:
    1. Extract sentiment score from Phase 2
    2. Extract technical score from Phase 3
    3. Calculate dynamic weights based on ADX
    4. Apply credibility multiplier
    5. Apply time factor
    6. Apply confluence bonus
    7. Calculate final score
    8. Build trade recommendation

    Attributes:
        credibility_manager: Manager for source credibility multipliers.
        time_calculator: Calculator for time-based factors.
        confluence_detector: Detector for signal confluence.
        weight_calculator: Calculator for dynamic sentiment/technical weights.
        recommendation_builder: Builder for trade recommendations.
    """

    def __init__(
        self,
        credibility_manager: SourceCredibilityManager,
        time_calculator: TimeFactorCalculator,
        confluence_detector: ConfluenceDetector,
        weight_calculator: DynamicWeightCalculator,
        recommendation_builder: RecommendationBuilder,
    ):
        """Initialize the SignalScorer with all required components.

        Args:
            credibility_manager: Manager for source credibility multipliers.
            time_calculator: Calculator for time-based factors.
            confluence_detector: Detector for signal confluence.
            weight_calculator: Calculator for dynamic sentiment/technical weights.
            recommendation_builder: Builder for trade recommendations.
        """
        self._credibility_manager = credibility_manager
        self._time_calculator = time_calculator
        self._confluence_detector = confluence_detector
        self._weight_calculator = weight_calculator
        self._recommendation_builder = recommendation_builder

    def score(
        self,
        validated_signal: ValidatedSignal,
        current_price: float,
        earnings_dates: dict[str, datetime] | None = None,
    ) -> TradeRecommendation:
        """Score a validated signal and produce trade recommendation.

        Pipeline:
        1. Extract sentiment score from Phase 2
        2. Extract technical score from Phase 3
        3. Calculate dynamic weights based on ADX
        4. Apply credibility multiplier
        5. Apply time factor
        6. Apply confluence bonus
        7. Calculate final score
        8. Build trade recommendation

        Args:
            validated_signal: The validated signal to score.
            current_price: Current price of the asset.
            earnings_dates: Optional dictionary mapping symbols to earnings dates.

        Returns:
            Complete TradeRecommendation with scoring details.
        """
        # Extract symbol from message tickers
        tickers = validated_signal.message.get_tickers()
        symbol = tickers[0] if tickers else "UNKNOWN"

        # Determine direction
        direction = self._determine_direction(validated_signal)

        # Handle VETO status - return NO_TRADE immediately
        if validated_signal.validation.status == ValidationStatus.VETO:
            components = ScoreComponents(
                sentiment_score=self._calculate_sentiment_score(validated_signal),
                technical_score=self._calculate_technical_score(validated_signal),
                sentiment_weight=0.5,
                technical_weight=0.5,
                confluence_bonus=0.0,
                credibility_multiplier=1.0,
                time_factor=1.0,
            )
            return self._recommendation_builder.build(
                symbol=symbol,
                direction=direction,
                score=0.0,
                current_price=current_price,
                components=components,
                reasoning="Signal vetoed by technical validation",
            )

        # Step 1 & 2: Calculate sentiment and technical scores
        sentiment_score = self._calculate_sentiment_score(validated_signal)
        technical_score = self._calculate_technical_score(validated_signal)

        # Step 3: Calculate dynamic weights based on ADX
        adx = validated_signal.validation.indicators.adx
        sentiment_weight, technical_weight = self._weight_calculator.calculate_weights(
            adx=adx
        )

        # Step 4: Get credibility multiplier
        author = validated_signal.message.message.author
        source = validated_signal.message.message.source
        credibility_multiplier = self._credibility_manager.get_multiplier(author, source)

        # Step 5: Calculate time factor
        timestamp = validated_signal.message.message.timestamp
        time_factor, _ = self._time_calculator.calculate_factor(
            timestamp=timestamp,
            symbol=symbol,
            earnings_dates=earnings_dates,
        )

        # Step 6: Get confluence bonus and record signal
        confluence_bonus = self._confluence_detector.get_bonus(symbol, timestamp)
        self._confluence_detector.record_signal(symbol, timestamp)

        # Step 7: Calculate final score using the formula
        # base_score = (sentiment_score * sentiment_weight) + (technical_score * technical_weight)
        # final_score = base_score * credibility_multiplier * time_factor * (1 + confluence_bonus)
        base_score = (sentiment_score * sentiment_weight) + (technical_score * technical_weight)
        final_score = base_score * credibility_multiplier * time_factor * (1 + confluence_bonus)

        # Clamp to 0-100
        final_score = min(100.0, max(0.0, final_score))

        # Create components
        components = ScoreComponents(
            sentiment_score=sentiment_score,
            technical_score=technical_score,
            sentiment_weight=sentiment_weight,
            technical_weight=technical_weight,
            confluence_bonus=confluence_bonus,
            credibility_multiplier=credibility_multiplier,
            time_factor=time_factor,
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            validated_signal=validated_signal,
            components=components,
            final_score=final_score,
        )

        # Step 8: Build trade recommendation
        return self._recommendation_builder.build(
            symbol=symbol,
            direction=direction,
            score=final_score,
            current_price=current_price,
            components=components,
            reasoning=reasoning,
        )

    def _calculate_sentiment_score(self, validated_signal: ValidatedSignal) -> float:
        """Convert sentiment analysis to 0-100 score.

        Args:
            validated_signal: The validated signal.

        Returns:
            Sentiment score from 0-100.
        """
        # sentiment.score is 0-1, convert to 0-100
        return validated_signal.message.sentiment.score * 100.0

    def _calculate_technical_score(self, validated_signal: ValidatedSignal) -> float:
        """Convert technical validation to 0-100 score.

        Technical Score Mapping:
        - PASS: 75 + (confidence_modifier - 1.0) * 50 (clamped 0-100)
        - VETO: 25 (penalty for veto)
        - WARN: 50

        Args:
            validated_signal: The validated signal.

        Returns:
            Technical score from 0-100.
        """
        validation = validated_signal.validation

        if validation.status == ValidationStatus.VETO:
            return 25.0

        if validation.status == ValidationStatus.WARN:
            return 50.0

        # PASS status
        # Base of 75 + adjustment based on confidence modifier
        score = 75.0 + (validation.confidence_modifier - 1.0) * 50.0
        # Clamp to 0-100
        return min(100.0, max(0.0, score))

    def _determine_direction(self, validated_signal: ValidatedSignal) -> Direction:
        """Determine trade direction from sentiment.

        Direction Mapping:
        - BULLISH -> LONG
        - BEARISH -> SHORT
        - NEUTRAL -> NEUTRAL

        Args:
            validated_signal: The validated signal.

        Returns:
            Trade direction based on sentiment label.
        """
        sentiment_label = validated_signal.message.sentiment.label

        if sentiment_label == SentimentLabel.BULLISH:
            return Direction.LONG
        elif sentiment_label == SentimentLabel.BEARISH:
            return Direction.SHORT
        else:
            return Direction.NEUTRAL

    def _generate_reasoning(
        self,
        validated_signal: ValidatedSignal,
        components: ScoreComponents,
        final_score: float,
    ) -> str:
        """Generate human-readable reasoning for the recommendation.

        Args:
            validated_signal: The validated signal.
            components: Score components.
            final_score: The final calculated score.

        Returns:
            Human-readable explanation string.
        """
        sentiment_label = validated_signal.message.sentiment.label.value
        validation_status = validated_signal.validation.status.value

        parts = [
            f"Sentiment: {sentiment_label} ({components.sentiment_score:.0f}/100)",
            f"Technical: {validation_status} ({components.technical_score:.0f}/100)",
        ]

        if components.credibility_multiplier != 1.0:
            parts.append(f"Credibility: {components.credibility_multiplier:.2f}x")

        if components.time_factor != 1.0:
            parts.append(f"Time factor: {components.time_factor:.2f}")

        if components.confluence_bonus > 0:
            parts.append(f"Confluence bonus: +{components.confluence_bonus:.0%}")

        parts.append(f"Final score: {final_score:.1f}/100")

        return " | ".join(parts)
