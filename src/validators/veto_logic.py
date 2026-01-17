# src/validators/veto_logic.py
"""Veto logic for technical signal validation.

This module implements the veto rules that can block trading signals
based on technical indicator conflicts.
"""

from src.analyzers.sentiment_result import SentimentLabel
from src.validators.models import TechnicalIndicators, ValidationStatus


class VetoLogic:
    """Applies veto rules to technical indicators.

    Veto rules prevent signals from being executed when technical
    indicators show conflicting or unfavorable conditions.

    Attributes:
        rsi_overbought: RSI threshold above which bullish signals are vetoed.
        rsi_oversold: RSI threshold below which bearish signals are vetoed.
        adx_trend_threshold: ADX threshold below which all signals are vetoed.
    """

    def __init__(
        self,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        adx_trend_threshold: float = 20.0,
    ):
        """Initialize veto logic with configurable thresholds.

        Args:
            rsi_overbought: RSI value above which market is considered overbought.
                Default is 70.0.
            rsi_oversold: RSI value below which market is considered oversold.
                Default is 30.0.
            adx_trend_threshold: ADX value below which market has insufficient trend.
                Default is 20.0.
        """
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.adx_trend_threshold = adx_trend_threshold

    def evaluate(
        self,
        indicators: TechnicalIndicators,
        sentiment: SentimentLabel,
    ) -> tuple[ValidationStatus, list[str]]:
        """Evaluate technical indicators against veto rules.

        Args:
            indicators: Technical indicators to evaluate.
            sentiment: Sentiment label (BULLISH, BEARISH, or NEUTRAL).

        Returns:
            Tuple of (ValidationStatus, list of veto reasons).
            Status is VETO if any rule triggers, PASS otherwise.
            Veto reasons list is empty if status is PASS.

        Veto Rules:
            1. BULLISH signal blocked if RSI > rsi_overbought
            2. BEARISH signal blocked if RSI < rsi_oversold
            3. ANY signal blocked if ADX < adx_trend_threshold (no trend)
            4. BULLISH blocked if macd_divergence == "bearish"
            5. BEARISH blocked if macd_divergence == "bullish"
            6. BULLISH blocked if macd_trend == "falling" AND macd_histogram < 0
            7. BEARISH blocked if macd_trend == "rising" AND macd_histogram > 0
        """
        # NEUTRAL sentiment bypasses all veto rules
        if sentiment == SentimentLabel.NEUTRAL:
            return ValidationStatus.PASS, []

        veto_reasons: list[str] = []

        # Rule 3: Check for sufficient trend strength (applies to all signals)
        if indicators.adx < self.adx_trend_threshold:
            veto_reasons.append(
                f"ADX ({indicators.adx:.1f}) below trend threshold "
                f"({self.adx_trend_threshold:.1f}) - insufficient trend strength"
            )

        # Apply sentiment-specific rules
        if sentiment == SentimentLabel.BULLISH:
            veto_reasons.extend(self._check_bullish_vetos(indicators))
        elif sentiment == SentimentLabel.BEARISH:
            veto_reasons.extend(self._check_bearish_vetos(indicators))

        # Determine status based on veto reasons
        if veto_reasons:
            return ValidationStatus.VETO, veto_reasons
        return ValidationStatus.PASS, []

    def _check_bullish_vetos(self, indicators: TechnicalIndicators) -> list[str]:
        """Check veto conditions specific to bullish signals.

        Args:
            indicators: Technical indicators to evaluate.

        Returns:
            List of veto reasons for bullish signals.
        """
        reasons: list[str] = []

        # Rule 1: RSI overbought check
        if indicators.rsi > self.rsi_overbought:
            reasons.append(
                f"RSI ({indicators.rsi:.1f}) is overbought "
                f"(>{self.rsi_overbought:.1f}) - bullish signal rejected"
            )

        # Rule 4: MACD bearish divergence check
        if indicators.macd_divergence == "bearish":
            reasons.append(
                "MACD divergence is bearish - conflicts with bullish signal"
            )

        # Rule 6: MACD falling with negative histogram
        if indicators.macd_trend == "falling" and indicators.macd_histogram < 0:
            reasons.append(
                f"MACD trend is falling with negative histogram "
                f"({indicators.macd_histogram:.3f}) - conflicts with bullish signal"
            )

        return reasons

    def _check_bearish_vetos(self, indicators: TechnicalIndicators) -> list[str]:
        """Check veto conditions specific to bearish signals.

        Args:
            indicators: Technical indicators to evaluate.

        Returns:
            List of veto reasons for bearish signals.
        """
        reasons: list[str] = []

        # Rule 2: RSI oversold check
        if indicators.rsi < self.rsi_oversold:
            reasons.append(
                f"RSI ({indicators.rsi:.1f}) is oversold "
                f"(<{self.rsi_oversold:.1f}) - bearish signal rejected"
            )

        # Rule 5: MACD bullish divergence check
        if indicators.macd_divergence == "bullish":
            reasons.append(
                "MACD divergence is bullish - conflicts with bearish signal"
            )

        # Rule 7: MACD rising with positive histogram
        if indicators.macd_trend == "rising" and indicators.macd_histogram > 0:
            reasons.append(
                f"MACD trend is rising with positive histogram "
                f"({indicators.macd_histogram:.3f}) - conflicts with bearish signal"
            )

        return reasons
