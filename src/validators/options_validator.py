# src/validators/options_validator.py
"""Validator for options flow data."""

from src.validators.models import OptionsFlowData


class OptionsValidator:
    """Validates options flow data and calculates confidence modifiers.

    This validator detects volume spikes, unusual options activity, and
    high implied volatility conditions that may affect signal quality.

    Attributes:
        volume_spike_ratio: Threshold for volume spike detection.
        iv_rank_warning_threshold: IV rank threshold for warnings.
    """

    def __init__(
        self,
        volume_spike_ratio: float = 2.0,
        iv_rank_warning_threshold: float = 80.0,
    ) -> None:
        """Initialize the options validator.

        Args:
            volume_spike_ratio: Volume ratio threshold for spike detection.
                Default is 2.0 (200% of average volume).
            iv_rank_warning_threshold: IV rank threshold above which to warn.
                Default is 80.0 (80th percentile).
        """
        self.volume_spike_ratio = volume_spike_ratio
        self.iv_rank_warning_threshold = iv_rank_warning_threshold

    def validate(self, data: OptionsFlowData) -> tuple[bool, list[str]]:
        """Validate options flow data.

        Checks for volume spikes and unusual activity that enhance signal
        quality, as well as high IV conditions that require warnings.

        Args:
            data: Options flow data to validate.

        Returns:
            A tuple of (is_enhanced, warnings) where:
            - is_enhanced: True if volume spike or unusual activity detected
            - warnings: List of warning messages
        """
        warnings: list[str] = []
        is_enhanced = False

        # Check for volume spike
        if data.volume_ratio >= self.volume_spike_ratio:
            is_enhanced = True

        # Check for unusual activity
        if data.unusual_activity:
            is_enhanced = True

        # Check for high IV (warning condition)
        if data.iv_rank > self.iv_rank_warning_threshold:
            warnings.append(
                f"High implied volatility detected (IV Rank: {data.iv_rank:.1f}). "
                "Options may be expensive."
            )

        return is_enhanced, warnings

    def get_confidence_modifier(self, data: OptionsFlowData) -> float:
        """Calculate confidence modifier based on options flow.

        The modifier adjusts signal confidence based on:
        - Volume spikes (+0.1)
        - Unusual activity (+0.1)
        - Low IV rank < 50 (+0.1, cheap options)
        - High IV rank > threshold (-0.2, expensive options)

        Args:
            data: Options flow data.

        Returns:
            Confidence modifier clamped to range [0.8, 1.3].
        """
        modifier = 1.0

        # Boost for volume spike
        if data.volume_ratio >= self.volume_spike_ratio:
            modifier += 0.1

        # Boost for unusual activity
        if data.unusual_activity:
            modifier += 0.1

        # Boost for low IV (cheap options)
        if data.iv_rank < 50.0:
            modifier += 0.1

        # Penalty for high IV (expensive options)
        if data.iv_rank > self.iv_rank_warning_threshold:
            modifier -= 0.2

        # Clamp to valid range [0.8, 1.3]
        return max(0.8, min(1.3, modifier))
