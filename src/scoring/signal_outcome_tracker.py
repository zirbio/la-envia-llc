# src/scoring/signal_outcome_tracker.py
"""Tracks signal outcomes for accuracy measurement."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.scoring.dynamic_credibility_manager import DynamicCredibilityManager

logger = logging.getLogger(__name__)


@dataclass
class PendingEvaluation:
    """A signal pending outcome evaluation."""

    signal_id: str
    author_id: str
    symbol: str
    direction: str  # "bullish" or "bearish"
    entry_price: float
    entry_time: datetime
    evaluate_at: datetime


class SignalOutcomeTracker:
    """Tracks signal outcomes to measure source accuracy.

    Records signals when trades are executed, then evaluates
    whether the prediction was correct after a time window.
    """

    def __init__(
        self,
        credibility_manager: DynamicCredibilityManager,
        alpaca_client,  # Type hint omitted to avoid circular import
        evaluation_window_minutes: int = 30,
        success_threshold_percent: float = 1.0,
    ):
        """Initialize the tracker.

        Args:
            credibility_manager: Manager to update with outcomes.
            alpaca_client: Alpaca client for price data.
            evaluation_window_minutes: Minutes to wait before evaluating.
            success_threshold_percent: Price change % to consider success.
        """
        self._credibility = credibility_manager
        self._alpaca = alpaca_client
        self._window_minutes = evaluation_window_minutes
        self._threshold_percent = success_threshold_percent
        self._pending_evaluations: list[PendingEvaluation] = []

    def record_signal(
        self,
        signal_id: str,
        author_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
    ) -> None:
        """Record a signal for future evaluation.

        Args:
            signal_id: Unique identifier for the signal.
            author_id: Author who generated the signal.
            symbol: Stock symbol.
            direction: "bullish" or "bearish".
            entry_price: Price at trade execution.
        """
        now = datetime.now()
        evaluation = PendingEvaluation(
            signal_id=signal_id,
            author_id=author_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=now,
            evaluate_at=now + timedelta(minutes=self._window_minutes),
        )
        self._pending_evaluations.append(evaluation)

        logger.info(
            f"Recorded signal {signal_id} from {author_id} for {symbol} "
            f"({direction}) @ ${entry_price:.2f}"
        )

    async def evaluate_pending(self) -> None:
        """Evaluate all signals that have passed the evaluation window."""
        now = datetime.now()
        ready = [e for e in self._pending_evaluations if e.evaluate_at <= now]

        for evaluation in ready:
            try:
                was_correct = await self._evaluate_outcome(evaluation)
                self._credibility.record_outcome(evaluation.author_id, was_correct)

                logger.info(
                    f"Evaluated signal {evaluation.signal_id}: "
                    f"{'CORRECT' if was_correct else 'INCORRECT'}"
                )
            except Exception as e:
                logger.error(f"Error evaluating {evaluation.signal_id}: {e}")
            finally:
                self._pending_evaluations.remove(evaluation)

    async def _evaluate_outcome(self, evaluation: PendingEvaluation) -> bool:
        """Determine if a signal prediction was correct.

        Args:
            evaluation: The pending evaluation to check.

        Returns:
            True if prediction was correct, False otherwise.
        """
        current_price = await self._alpaca.get_current_price(evaluation.symbol)
        price_change_pct = (
            (current_price - evaluation.entry_price) / evaluation.entry_price
        ) * 100

        if evaluation.direction == "bullish":
            return price_change_pct >= self._threshold_percent
        else:  # bearish
            return price_change_pct <= -self._threshold_percent
