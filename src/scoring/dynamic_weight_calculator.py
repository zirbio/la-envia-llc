# src/scoring/dynamic_weight_calculator.py
"""Dynamic weight calculator for scoring system."""


class DynamicWeightCalculator:
    """Calculates sentiment/technical weights based on market conditions.

    The calculator adjusts the balance between sentiment and technical
    analysis based on market trend strength (ADX) and volatility.

    Weight Rules:
        - Strong trend (ADX > 30): Favor technical (0.4 sentiment, 0.6 technical)
        - Normal trend (20 <= ADX <= 30): Balanced (0.5, 0.5)
        - Weak trend (ADX < 20): Favor sentiment (0.6 sentiment, 0.4 technical)
        - High volatility (percentile > 80): Override to (0.35, 0.65)

    Attributes:
        base_sentiment_weight: Default sentiment weight for normal trend.
        base_technical_weight: Default technical weight for normal trend.
        strong_trend_adx: ADX threshold for strong trend.
        weak_trend_adx: ADX threshold for weak trend.
        strong_trend_technical_weight: Technical weight during strong trends.
        weak_trend_sentiment_weight: Sentiment weight during weak trends.
    """

    # High volatility override weights
    HIGH_VOLATILITY_SENTIMENT_WEIGHT = 0.35
    HIGH_VOLATILITY_TECHNICAL_WEIGHT = 0.65
    HIGH_VOLATILITY_THRESHOLD = 80.0

    def __init__(
        self,
        base_sentiment_weight: float = 0.5,
        base_technical_weight: float = 0.5,
        strong_trend_adx: float = 30.0,
        weak_trend_adx: float = 20.0,
        strong_trend_technical_weight: float = 0.6,
        weak_trend_sentiment_weight: float = 0.6,
    ):
        """Initialize the dynamic weight calculator.

        Args:
            base_sentiment_weight: Default sentiment weight for normal trend.
            base_technical_weight: Default technical weight for normal trend.
            strong_trend_adx: ADX threshold above which trend is considered strong.
            weak_trend_adx: ADX threshold below which trend is considered weak.
            strong_trend_technical_weight: Technical weight when ADX > strong_trend_adx.
            weak_trend_sentiment_weight: Sentiment weight when ADX < weak_trend_adx.
        """
        self._base_sentiment_weight = base_sentiment_weight
        self._base_technical_weight = base_technical_weight
        self._strong_trend_adx = strong_trend_adx
        self._weak_trend_adx = weak_trend_adx
        self._strong_trend_technical_weight = strong_trend_technical_weight
        self._weak_trend_sentiment_weight = weak_trend_sentiment_weight

    def calculate_weights(
        self,
        adx: float,
        volatility_percentile: float = 50.0,
    ) -> tuple[float, float]:
        """Calculate sentiment and technical weights based on market conditions.

        Args:
            adx: Average Directional Index value (trend strength indicator).
            volatility_percentile: Volatility percentile (0-100). Values > 80
                trigger a high volatility override.

        Returns:
            Tuple of (sentiment_weight, technical_weight) that sum to 1.0.
        """
        # High volatility override takes precedence
        if volatility_percentile > self.HIGH_VOLATILITY_THRESHOLD:
            return (
                self.HIGH_VOLATILITY_SENTIMENT_WEIGHT,
                self.HIGH_VOLATILITY_TECHNICAL_WEIGHT,
            )

        # Strong trend: favor technical analysis
        if adx > self._strong_trend_adx:
            technical_weight = self._strong_trend_technical_weight
            sentiment_weight = 1.0 - technical_weight
            return (sentiment_weight, technical_weight)

        # Weak trend: favor sentiment analysis
        if adx < self._weak_trend_adx:
            sentiment_weight = self._weak_trend_sentiment_weight
            technical_weight = 1.0 - sentiment_weight
            return (sentiment_weight, technical_weight)

        # Normal trend: use base weights
        return (self._base_sentiment_weight, self._base_technical_weight)
