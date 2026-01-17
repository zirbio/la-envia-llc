# src/journal/metrics_calculator.py
"""Calculator for trading performance metrics."""
import math

from src.journal.models import JournalEntry, TradingMetrics


class MetricsCalculator:
    """Calculates trading performance metrics from journal entries."""

    def calculate(
        self,
        entries: list[JournalEntry],
        period_days: int = 30,
    ) -> TradingMetrics:
        """Calculate trading metrics from journal entries.

        Args:
            entries: List of journal entries to analyze.
            period_days: Number of days the metrics cover.

        Returns:
            TradingMetrics with all calculated values.
        """
        closed_entries = [e for e in entries if not e.is_open]

        if not closed_entries:
            return self._empty_metrics(period_days)

        winners = [e for e in closed_entries if e.pnl_dollars > 0]
        losers = [e for e in closed_entries if e.pnl_dollars < 0]

        total_trades = len(closed_entries)
        winning_trades = len(winners)
        losing_trades = len(losers)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(e.pnl_dollars for e in winners)
        gross_loss = abs(sum(e.pnl_dollars for e in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        avg_win_dollars = gross_profit / winning_trades if winning_trades > 0 else 0.0
        avg_loss_dollars = gross_loss / losing_trades if losing_trades > 0 else 0.0

        avg_win_r = sum(e.r_multiple for e in winners) / winning_trades if winning_trades > 0 else 0.0
        avg_loss_r = abs(sum(e.r_multiple for e in losers)) / losing_trades if losing_trades > 0 else 0.0

        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

        total_pnl_dollars = sum(e.pnl_dollars for e in closed_entries)
        total_pnl_percent = sum(e.pnl_percent for e in closed_entries)

        max_drawdown_percent = self._calculate_max_drawdown(closed_entries)
        sharpe_ratio = self._calculate_sharpe_ratio(closed_entries)

        best_trade = max(closed_entries, key=lambda e: e.pnl_dollars)
        worst_trade = min(closed_entries, key=lambda e: e.pnl_dollars)

        return TradingMetrics(
            period_days=period_days,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win_dollars=avg_win_dollars,
            avg_loss_dollars=avg_loss_dollars,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            total_pnl_dollars=total_pnl_dollars,
            total_pnl_percent=total_pnl_percent,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )

    def _empty_metrics(self, period_days: int) -> TradingMetrics:
        """Return metrics with zero values for empty entry list."""
        return TradingMetrics(
            period_days=period_days,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_win_dollars=0.0,
            avg_loss_dollars=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            total_pnl_dollars=0.0,
            total_pnl_percent=0.0,
            max_drawdown_percent=0.0,
            sharpe_ratio=0.0,
            best_trade=None,
            worst_trade=None,
        )

    def _calculate_max_drawdown(self, entries: list[JournalEntry]) -> float:
        """Calculate maximum drawdown from cumulative PnL.

        Args:
            entries: List of closed journal entries.

        Returns:
            Maximum drawdown as a percentage.
        """
        if not entries:
            return 0.0

        sorted_entries = sorted(entries, key=lambda e: e.exit_time or e.entry_time)
        cumulative_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for entry in sorted_entries:
            cumulative_pnl += entry.pnl_percent
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown

    def _calculate_sharpe_ratio(self, entries: list[JournalEntry]) -> float:
        """Calculate Sharpe ratio from returns.

        Args:
            entries: List of closed journal entries.

        Returns:
            Annualized Sharpe ratio.
        """
        if len(entries) < 2:
            return 0.0

        returns = [e.pnl_percent for e in entries]
        avg_return = sum(returns) / len(returns)

        variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        annualization_factor = math.sqrt(252)
        return (avg_return / std_dev) * annualization_factor
