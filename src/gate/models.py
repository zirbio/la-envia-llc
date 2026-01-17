"""Data models for market condition gate."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class GateCheckResult:
    """Result of a single gate check.

    Attributes:
        name: Check identifier (e.g., "trading_hours", "volume", "vix", "choppy").
        passed: Whether the check passed.
        reason: Explanation if check failed, None if passed.
        data: Observed data values used in the check.
    """

    name: str
    passed: bool
    reason: str | None
    data: dict[str, Any]


@dataclass
class GateStatus:
    """Combined result of all gate checks.

    Attributes:
        timestamp: When the gate check was performed.
        is_open: Whether all checks passed and trading is allowed.
        checks: List of individual check results.
        position_size_factor: Multiplier for position size (1.0 normal, 0.5 elevated, 0.0 blocked).
    """

    timestamp: datetime
    is_open: bool
    checks: list[GateCheckResult]
    position_size_factor: float

    def get_failed_checks(self) -> list[GateCheckResult]:
        """Return checks that did not pass.

        Returns:
            List of GateCheckResult where passed is False.
        """
        return [check for check in self.checks if not check.passed]
