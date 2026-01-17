"""Data models for risk management."""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class RiskCheckResult:
    """Result of a risk check for a proposed trade.

    Attributes:
        approved: Whether the trade is approved to proceed.
        adjusted_quantity: Final quantity after risk adjustments (0 if blocked).
        adjusted_value: Final dollar value after risk adjustments.
        rejection_reason: Explanation if trade was rejected (None if approved).
        warnings: List of non-blocking warnings about the trade.
    """

    approved: bool
    adjusted_quantity: int
    adjusted_value: float
    rejection_reason: str | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DailyRiskState:
    """Current risk state for a trading day.

    Attributes:
        date: The trading date for this state.
        realized_pnl: Sum of profit/loss from closed positions.
        unrealized_pnl: Current profit/loss from open positions.
        trades_today: Number of trades executed today.
        is_blocked: Whether trading is blocked due to hitting limits.
    """

    date: date
    realized_pnl: float
    unrealized_pnl: float
    trades_today: int
    is_blocked: bool
