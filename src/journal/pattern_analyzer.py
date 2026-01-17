# src/journal/pattern_analyzer.py
"""Analyzer for identifying trading patterns."""
from collections import defaultdict

from src.journal.models import JournalEntry, PatternAnalysis


class PatternAnalyzer:
    """Analyzes trading patterns from journal entries."""

    def analyze(self, entries: list[JournalEntry]) -> PatternAnalysis:
        """Analyze trading patterns from journal entries.

        Args:
            entries: List of journal entries to analyze.

        Returns:
            PatternAnalysis with identified patterns.
        """
        closed_entries = [e for e in entries if not e.is_open]

        if not closed_entries:
            return self._empty_analysis()

        best_hour, worst_hour = self._analyze_hours(closed_entries)
        best_day_of_week = self._analyze_days(closed_entries)
        best_symbols, worst_symbols = self._analyze_symbols(closed_entries)
        best_setups, worst_setups = self._analyze_setups(closed_entries)
        avg_winner_duration, avg_loser_duration = self._analyze_durations(closed_entries)

        return PatternAnalysis(
            best_hour=best_hour,
            worst_hour=worst_hour,
            best_day_of_week=best_day_of_week,
            best_symbols=best_symbols,
            worst_symbols=worst_symbols,
            best_setups=best_setups,
            worst_setups=worst_setups,
            avg_winner_duration_minutes=avg_winner_duration,
            avg_loser_duration_minutes=avg_loser_duration,
        )

    def _empty_analysis(self) -> PatternAnalysis:
        """Return analysis with default values for empty entry list."""
        return PatternAnalysis(
            best_hour=-1,
            worst_hour=-1,
            best_day_of_week=-1,
            best_symbols=[],
            worst_symbols=[],
            best_setups=[],
            worst_setups=[],
            avg_winner_duration_minutes=0.0,
            avg_loser_duration_minutes=0.0,
        )

    def _analyze_hours(
        self, entries: list[JournalEntry]
    ) -> tuple[int, int]:
        """Analyze trading performance by hour.

        Args:
            entries: List of closed journal entries.

        Returns:
            Tuple of (best_hour, worst_hour) based on avg PnL.
        """
        hour_pnl: dict[int, list[float]] = defaultdict(list)

        for entry in entries:
            hour = entry.entry_time.hour
            hour_pnl[hour].append(entry.pnl_dollars)

        hour_avg = {
            hour: sum(pnls) / len(pnls)
            for hour, pnls in hour_pnl.items()
        }

        best_hour = max(hour_avg, key=lambda h: hour_avg[h])
        worst_hour = min(hour_avg, key=lambda h: hour_avg[h])

        return best_hour, worst_hour

    def _analyze_days(self, entries: list[JournalEntry]) -> int:
        """Analyze trading performance by day of week.

        Args:
            entries: List of closed journal entries.

        Returns:
            Best day of week (0=Monday, 4=Friday) based on avg PnL.
        """
        day_pnl: dict[int, list[float]] = defaultdict(list)

        for entry in entries:
            day = entry.entry_time.weekday()
            day_pnl[day].append(entry.pnl_dollars)

        day_avg = {
            day: sum(pnls) / len(pnls)
            for day, pnls in day_pnl.items()
        }

        return max(day_avg, key=lambda d: day_avg[d])

    def _analyze_symbols(
        self, entries: list[JournalEntry]
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Analyze trading performance by symbol.

        Args:
            entries: List of closed journal entries.

        Returns:
            Tuple of (best_symbols, worst_symbols) as lists of (symbol, avg_pnl).
        """
        symbol_pnl: dict[str, list[float]] = defaultdict(list)

        for entry in entries:
            symbol_pnl[entry.symbol].append(entry.pnl_dollars)

        symbol_avg = [
            (symbol, sum(pnls) / len(pnls))
            for symbol, pnls in symbol_pnl.items()
        ]

        sorted_by_pnl = sorted(symbol_avg, key=lambda x: x[1], reverse=True)
        best_symbols = sorted_by_pnl[:5]
        worst_symbols = sorted(symbol_avg, key=lambda x: x[1])[:5]

        return best_symbols, worst_symbols

    def _analyze_setups(
        self, entries: list[JournalEntry]
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Analyze trading performance by setup (entry_reason).

        Args:
            entries: List of closed journal entries.

        Returns:
            Tuple of (best_setups, worst_setups) as lists of (setup, avg_pnl).
        """
        setup_pnl: dict[str, list[float]] = defaultdict(list)

        for entry in entries:
            setup_pnl[entry.entry_reason].append(entry.pnl_dollars)

        setup_avg = [
            (setup, sum(pnls) / len(pnls))
            for setup, pnls in setup_pnl.items()
        ]

        sorted_by_pnl = sorted(setup_avg, key=lambda x: x[1], reverse=True)
        best_setups = sorted_by_pnl[:5]
        worst_setups = sorted(setup_avg, key=lambda x: x[1])[:5]

        return best_setups, worst_setups

    def _analyze_durations(
        self, entries: list[JournalEntry]
    ) -> tuple[float, float]:
        """Analyze trade durations for winners and losers.

        Args:
            entries: List of closed journal entries.

        Returns:
            Tuple of (avg_winner_duration_minutes, avg_loser_duration_minutes).
        """
        winner_durations: list[float] = []
        loser_durations: list[float] = []

        for entry in entries:
            if entry.exit_time is None:
                continue

            duration_seconds = (entry.exit_time - entry.entry_time).total_seconds()
            duration_minutes = duration_seconds / 60.0

            if entry.pnl_dollars > 0:
                winner_durations.append(duration_minutes)
            elif entry.pnl_dollars < 0:
                loser_durations.append(duration_minutes)

        avg_winner = sum(winner_durations) / len(winner_durations) if winner_durations else 0.0
        avg_loser = sum(loser_durations) / len(loser_durations) if loser_durations else 0.0

        return avg_winner, avg_loser
