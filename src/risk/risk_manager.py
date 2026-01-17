"""Risk management engine for validating trades."""

from datetime import date

from risk.models import DailyRiskState, RiskCheckResult
from scoring.models import Direction, TradeRecommendation


class RiskManager:
    """Manages risk checks and position limits for trading.

    Validates trades against position limits, daily loss limits, and risk thresholds.
    Maintains daily state to track realized/unrealized PnL and enforce daily limits.

    Attributes:
        max_position_value: Maximum dollar value allowed for a single position.
        max_daily_loss: Maximum daily loss before blocking further trades.
        unrealized_warning_threshold: Threshold for unrealized loss warnings.
    """

    def __init__(
        self,
        max_position_value: float,
        max_daily_loss: float,
        unrealized_warning_threshold: float = 300.0,
    ):
        """Initialize RiskManager with configuration parameters.

        Args:
            max_position_value: Maximum dollar value for a single position.
            max_daily_loss: Maximum daily loss before blocking trades.
            unrealized_warning_threshold: Threshold for unrealized loss warnings.
                Defaults to 300.0.
        """
        self.max_position_value = max_position_value
        self.max_daily_loss = max_daily_loss
        self.unrealized_warning_threshold = unrealized_warning_threshold

        # Initialize daily state with today's date and zero values
        self._daily_state = DailyRiskState(
            date=date.today(),
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            trades_today=0,
            is_blocked=False,
        )

    def check_trade(
        self,
        recommendation: TradeRecommendation,
        requested_quantity: int,
        current_price: float,
    ) -> RiskCheckResult:
        """Validate trade against risk rules.

        Performs the following checks in order:
        1. Rejects NEUTRAL direction trades
        2. Rejects trades if daily trading is blocked
        3. Rejects trades exceeding max position value
        4. Warns if unrealized PnL exceeds warning threshold

        Args:
            recommendation: The trade recommendation to validate.
            requested_quantity: Number of shares/contracts requested.
            current_price: Current market price for the asset.

        Returns:
            RiskCheckResult with approval status, adjusted quantities, and any
            warnings or rejection reasons.
        """
        # Check 1: Reject NEUTRAL direction
        if recommendation.direction == Direction.NEUTRAL:
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0,
                adjusted_value=0.0,
                rejection_reason="Cannot execute trade with NEUTRAL direction",
                warnings=[],
            )

        # Check 2: Reject if daily trading is blocked
        if self._daily_state.is_blocked:
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0,
                adjusted_value=0.0,
                rejection_reason="Trading blocked: daily loss limit exceeded",
                warnings=[],
            )

        # Calculate position value
        position_value = requested_quantity * current_price

        # Check 3: Reject if position value exceeds limit
        if position_value > self.max_position_value:
            return RiskCheckResult(
                approved=False,
                adjusted_quantity=0,
                adjusted_value=0.0,
                rejection_reason=f"Position value (${position_value:.2f}) exceeds maximum allowed (${self.max_position_value:.2f})",
                warnings=[],
            )

        # Check 4: Warn if unrealized drawdown exceeds threshold (but don't block)
        warnings = []
        unrealized_loss = abs(self._daily_state.unrealized_pnl)
        if (
            self._daily_state.unrealized_pnl < 0
            and unrealized_loss >= self.unrealized_warning_threshold
        ):
            warnings.append(
                f"Unrealized drawdown (${unrealized_loss:.2f}) exceeds warning threshold (${self.unrealized_warning_threshold:.2f})"
            )

        # All checks passed - approve the trade
        return RiskCheckResult(
            approved=True,
            adjusted_quantity=requested_quantity,
            adjusted_value=position_value,
            rejection_reason=None,
            warnings=warnings,
        )
