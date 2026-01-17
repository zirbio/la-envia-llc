# src/journal/journal_manager.py
"""Manager for orchestrating all journal components."""
from datetime import date, datetime, timedelta

from src.execution.models import TrackedPosition
from src.journal.metrics_calculator import MetricsCalculator
from src.journal.models import JournalEntry
from src.journal.pattern_analyzer import PatternAnalyzer
from src.journal.settings import JournalSettings
from src.journal.trade_logger import TradeLogger
from src.scoring.models import TradeRecommendation


class JournalManager:
    """Orchestrates all journal components for trade logging and analysis.

    Coordinates TradeLogger, MetricsCalculator, and PatternAnalyzer to provide
    a unified interface for trade journaling and performance analysis.
    """

    def __init__(self, settings: JournalSettings) -> None:
        """Initialize the journal manager with all components.

        Args:
            settings: Journal configuration settings.
        """
        self._settings = settings
        self._logger = TradeLogger(settings)
        self._metrics_calculator = MetricsCalculator()
        self._pattern_analyzer = PatternAnalyzer()

    async def on_trade_opened(
        self, position: TrackedPosition, recommendation: TradeRecommendation
    ) -> str | None:
        """Log a trade entry when a position is opened.

        Args:
            position: The tracked position that was opened.
            recommendation: The trade recommendation that triggered the entry.

        Returns:
            The generated trade ID, or None if journaling is disabled.
        """
        if not self._settings.enabled:
            return None

        return await self._logger.log_entry(position, recommendation)

    async def on_trade_closed(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        exit_quantity: int,
        exit_reason: str,
    ) -> None:
        """Log a trade exit when a position is closed.

        Args:
            trade_id: The trade ID to update.
            exit_time: When the position was closed.
            exit_price: The exit price.
            exit_quantity: Number of shares exited.
            exit_reason: Reason for exiting.
        """
        if not self._settings.enabled:
            return

        await self._logger.log_exit(
            trade_id=trade_id,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_quantity=exit_quantity,
            exit_reason=exit_reason,
        )

    async def add_emotion_tag(self, trade_id: str, tag: str) -> None:
        """Add an emotion tag to a trade entry.

        Args:
            trade_id: The trade ID to update.
            tag: The emotion tag to add.
        """
        if not self._settings.enabled:
            return

        await self._logger.add_emotion_tag(trade_id, tag)

    async def add_notes(self, trade_id: str, notes: str) -> None:
        """Add notes to a trade entry.

        Args:
            trade_id: The trade ID to update.
            notes: The notes to add.
        """
        if not self._settings.enabled:
            return

        await self._logger.add_notes(trade_id, notes)

    async def get_entries_for_date(self, query_date: date) -> list[JournalEntry]:
        """Get all journal entries for a specific date.

        Args:
            query_date: The date to retrieve entries for.

        Returns:
            List of JournalEntry objects for that date.
        """
        return await self._logger.get_entries_for_date(query_date)

    async def get_daily_summary(self, query_date: date) -> dict:
        """Get a summary of trading activity for a specific date.

        Args:
            query_date: The date to summarize.

        Returns:
            Dict with date, total_trades, winning_trades, losing_trades,
            total_pnl, and win_rate.
        """
        entries = await self._logger.get_entries_for_date(query_date)
        closed_entries = [e for e in entries if not e.is_open]

        winning_trades = [e for e in closed_entries if e.pnl_dollars > 0]
        losing_trades = [e for e in closed_entries if e.pnl_dollars < 0]

        total_trades = len(closed_entries)
        total_pnl = sum(e.pnl_dollars for e in closed_entries)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        return {
            "date": query_date,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
        }

    async def get_weekly_report(self, end_date: date) -> dict:
        """Get a weekly performance report.

        Args:
            end_date: The end date of the week to report on.

        Returns:
            Dict with start_date, end_date, metrics (TradingMetrics),
            and patterns (PatternAnalysis).
        """
        start_date = end_date - timedelta(days=6)

        entries = await self._logger.get_entries_for_period(start_date, end_date)

        metrics = self._metrics_calculator.calculate(entries, period_days=7)
        patterns = self._pattern_analyzer.analyze(entries)

        return {
            "start_date": start_date,
            "end_date": end_date,
            "metrics": metrics,
            "patterns": patterns,
        }
