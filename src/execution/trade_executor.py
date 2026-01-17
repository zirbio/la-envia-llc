# src/execution/trade_executor.py
"""Trade executor for managing order execution and position tracking."""
from datetime import datetime

from src.execution.alpaca_client import AlpacaClient
from src.execution.models import ExecutionResult, TrackedPosition
from src.risk.models import RiskCheckResult
from src.risk.risk_manager import RiskManager
from src.scoring.models import Direction, TradeRecommendation


class TradeExecutor:
    """Execute trades and track positions.

    Coordinates order execution via Alpaca, maintains position tracking,
    and updates risk management state.

    Attributes:
        _alpaca: Alpaca client for order execution.
        _risk_manager: Risk manager for tracking trades and PnL.
        _tracked_positions: Dictionary of tracked positions keyed by symbol.
    """

    def __init__(
        self,
        alpaca_client: AlpacaClient,
        risk_manager: RiskManager,
    ):
        """Initialize TradeExecutor.

        Args:
            alpaca_client: Alpaca client for order execution.
            risk_manager: Risk manager for position tracking.
        """
        self._alpaca = alpaca_client
        self._risk_manager = risk_manager
        self._tracked_positions: dict[str, TrackedPosition] = {}

    async def execute(
        self,
        recommendation: TradeRecommendation,
        risk_result: RiskCheckResult,
    ) -> ExecutionResult:
        """Execute an approved trade.

        Steps:
        1. Check if risk_result.approved is True, else return failure
        2. Call sync_positions() (placeholder for now, returns [])
        3. Submit market order via alpaca_client.submit_order()
        4. Track position in _tracked_positions
        5. Call risk_manager.record_trade() and update_unrealized_pnl()
        6. Return ExecutionResult

        Args:
            recommendation: Trade recommendation to execute.
            risk_result: Result of risk check for the trade.

        Returns:
            ExecutionResult containing execution details.
        """
        timestamp = datetime.now()

        # Step 1: Check if trade is approved
        if not risk_result.approved:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=recommendation.symbol,
                side=self._direction_to_side(recommendation.direction),
                quantity=0,
                filled_price=None,
                error_message=risk_result.rejection_reason,
                timestamp=timestamp,
            )

        # Step 2: Sync positions (placeholder - returns empty list)
        await self.sync_positions()

        # Step 3: Submit market order via Alpaca
        try:
            side = self._direction_to_side(recommendation.direction)
            order = await self._alpaca.submit_order(
                symbol=recommendation.symbol,
                qty=risk_result.adjusted_quantity,
                side=side,
                order_type="market",
            )

            # Extract order details
            order_id = order["id"]
            filled_price = order.get("filled_avg_price")
            filled_qty = order.get("filled_qty", risk_result.adjusted_quantity)

            # Step 4: Track position
            tracked_position = TrackedPosition(
                symbol=recommendation.symbol,
                quantity=filled_qty,
                entry_price=filled_price,
                entry_time=timestamp,
                stop_loss=recommendation.stop_loss,
                take_profit=recommendation.take_profit,
                order_id=order_id,
                direction=recommendation.direction,
            )
            self._tracked_positions[recommendation.symbol] = tracked_position

            # Step 5: Update risk manager
            self._risk_manager.record_trade(
                recommendation.symbol,
                filled_qty,
                filled_price,
            )
            # Get unrealized PnL (placeholder returns 0.0)
            unrealized_pnl = await self.get_unrealized_pnl()
            self._risk_manager.update_unrealized_pnl(unrealized_pnl)

            # Step 6: Return success result
            return ExecutionResult(
                success=True,
                order_id=order_id,
                symbol=recommendation.symbol,
                side=side,
                quantity=filled_qty,
                filled_price=filled_price,
                error_message=None,
                timestamp=timestamp,
            )

        except Exception as e:
            # Handle execution errors
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=recommendation.symbol,
                side=self._direction_to_side(recommendation.direction),
                quantity=0,
                filled_price=None,
                error_message=str(e),
                timestamp=timestamp,
            )

    def _direction_to_side(self, direction: Direction) -> str:
        """Convert Direction to 'buy' or 'sell'.

        Args:
            direction: Trade direction (LONG or SHORT).

        Returns:
            "buy" for LONG, "sell" for SHORT.
        """
        return "buy" if direction == Direction.LONG else "sell"

    async def sync_positions(self) -> list[str]:
        """Compare tracked positions with Alpaca, detect closes.

        Returns list of symbols that were closed.
        For each closed position:
        1. Fetch current positions from Alpaca
        2. Find tracked positions no longer in Alpaca
        3. Calculate realized P&L
        4. Call risk_manager.record_close(symbol, pnl)
        5. Remove from _tracked_positions

        Returns:
            List of symbols that were closed.
        """
        alpaca_positions = await self._alpaca.get_all_positions()
        alpaca_symbols = {p["symbol"] for p in alpaca_positions}

        closed_symbols = []
        for symbol, tracked in list(self._tracked_positions.items()):
            if symbol not in alpaca_symbols:
                closed_symbols.append(symbol)
                pnl = await self._calculate_closed_pnl(tracked)
                self._risk_manager.record_close(symbol, pnl)
                del self._tracked_positions[symbol]

        return closed_symbols

    async def _calculate_closed_pnl(self, position: TrackedPosition) -> float:
        """Calculate realized P&L for closed position.

        For LONG: (exit_price - entry_price) * quantity
        For SHORT: (entry_price - exit_price) * quantity

        Try to get exit price from order history, fallback to take_profit.

        Args:
            position: The tracked position that was closed.

        Returns:
            Realized profit/loss as a float.
        """
        try:
            order = await self._alpaca.get_order(position.order_id)
            exit_price = order.get("filled_avg_price", position.take_profit)
        except Exception:
            exit_price = position.take_profit

        if position.direction == Direction.LONG:
            return (exit_price - position.entry_price) * position.quantity
        else:
            return (position.entry_price - exit_price) * position.quantity

    async def get_unrealized_pnl(self) -> float:
        """Sum unrealized P&L from all Alpaca positions.

        Returns:
            Total unrealized profit/loss across all positions.
        """
        positions = await self._alpaca.get_all_positions()
        return sum(float(p.get("unrealized_pl", 0)) for p in positions)

    def get_tracked_positions(self) -> dict[str, TrackedPosition]:
        """Return copy of tracked positions.

        Returns:
            Dictionary of tracked positions keyed by symbol.
        """
        return self._tracked_positions.copy()
