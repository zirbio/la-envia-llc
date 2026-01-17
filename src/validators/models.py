# src/validators/models.py
"""Data models for technical validation of trading signals."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.analyzers.analyzed_message import AnalyzedMessage


class ValidationStatus(Enum):
    """Status of technical validation."""

    PASS = "pass"
    VETO = "veto"
    WARN = "warn"


@dataclass
class TechnicalIndicators:
    """Technical indicators for validation.

    Attributes:
        rsi: Relative Strength Index (0-100).
        macd_histogram: MACD histogram value.
        macd_trend: MACD trend direction ("rising" | "falling" | "flat").
        stochastic_k: Stochastic %K value (0-100).
        stochastic_d: Stochastic %D value (0-100).
        adx: Average Directional Index (0-100).
        rsi_divergence: Optional RSI divergence signal.
        macd_divergence: Optional MACD divergence signal.
    """

    rsi: float
    macd_histogram: float
    macd_trend: str
    stochastic_k: float
    stochastic_d: float
    adx: float
    rsi_divergence: Optional[str] = None
    macd_divergence: Optional[str] = None


@dataclass
class OptionsFlowData:
    """Options flow data for validation.

    Attributes:
        volume_ratio: Options volume ratio vs average.
        iv_rank: Implied volatility rank (0-100).
        put_call_ratio: Put/call ratio.
        unusual_activity: Whether unusual options activity detected.
    """

    volume_ratio: float
    iv_rank: float
    put_call_ratio: float
    unusual_activity: bool


@dataclass
class TechnicalValidation:
    """Technical validation result.

    Attributes:
        status: Validation status (PASS, VETO, or WARN).
        indicators: Technical indicators used for validation.
        options_flow: Optional options flow data.
        veto_reasons: List of reasons for veto (if status is VETO).
        warnings: List of warning messages (if status is WARN).
        confidence_modifier: Confidence adjustment factor (0.0-1.0+).
    """

    status: ValidationStatus
    indicators: TechnicalIndicators
    confidence_modifier: float
    options_flow: Optional[OptionsFlowData] = None
    veto_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidatedSignal:
    """A trading signal with technical validation.

    Attributes:
        message: The analyzed message from Phase 2.
        validation: Technical validation result.
    """

    message: AnalyzedMessage
    validation: TechnicalValidation

    def should_trade(self) -> bool:
        """Determine if this signal should be traded.

        Returns:
            True if status is not VETO, False otherwise.
        """
        return self.validation.status != ValidationStatus.VETO
