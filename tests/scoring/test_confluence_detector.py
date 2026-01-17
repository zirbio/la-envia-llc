# tests/scoring/test_confluence_detector.py
"""Tests for confluence detector."""
from datetime import datetime, timedelta

import pytest

from src.scoring.confluence_detector import ConfluenceDetector


class TestConfluenceDetector:
    """Tests for ConfluenceDetector."""

    def test_single_signal_no_bonus(self):
        """Test that one signal should return 0.0 bonus."""
        detector = ConfluenceDetector()
        now = datetime.now()

        detector.record_signal("AAPL", now)
        bonus = detector.get_bonus("AAPL", now)

        assert bonus == 0.0

    def test_two_signals_within_window_returns_bonus(self):
        """Test that two signals within 15 min returns 0.10 bonus."""
        detector = ConfluenceDetector(window_minutes=15)
        now = datetime.now()

        # First signal 10 minutes ago
        detector.record_signal("AAPL", now - timedelta(minutes=10))
        # Second signal now
        detector.record_signal("AAPL", now)

        bonus = detector.get_bonus("AAPL", now)

        assert bonus == 0.10

    def test_three_signals_returns_higher_bonus(self):
        """Test that three or more signals returns 0.20 bonus."""
        detector = ConfluenceDetector(window_minutes=15)
        now = datetime.now()

        # Three signals within window
        detector.record_signal("AAPL", now - timedelta(minutes=10))
        detector.record_signal("AAPL", now - timedelta(minutes=5))
        detector.record_signal("AAPL", now)

        bonus = detector.get_bonus("AAPL", now)

        assert bonus == 0.20

    def test_signals_outside_window_not_counted(self):
        """Test that signal 20 min ago should not count."""
        detector = ConfluenceDetector(window_minutes=15)
        now = datetime.now()

        # Signal 20 minutes ago (outside 15-minute window)
        detector.record_signal("AAPL", now - timedelta(minutes=20))
        # Signal now
        detector.record_signal("AAPL", now)

        # Only 1 signal within window, so no bonus
        bonus = detector.get_bonus("AAPL", now)

        assert bonus == 0.0

    def test_cleanup_removes_old_signals(self):
        """Test that cleanup should remove old signals."""
        detector = ConfluenceDetector(window_minutes=15)
        now = datetime.now()

        # Record signals - one old, one recent
        detector.record_signal("AAPL", now - timedelta(minutes=20))
        detector.record_signal("AAPL", now - timedelta(minutes=5))

        # Cleanup old signals
        detector.cleanup_old_signals()

        # Only the recent signal should remain
        count = detector.get_confluence_count("AAPL", now)

        assert count == 1

    def test_get_confluence_count_returns_correct_count(self):
        """Test that get_confluence_count returns correct number of signals."""
        detector = ConfluenceDetector(window_minutes=15)
        now = datetime.now()

        # Record 3 signals within window
        detector.record_signal("AAPL", now - timedelta(minutes=10))
        detector.record_signal("AAPL", now - timedelta(minutes=5))
        detector.record_signal("AAPL", now)

        count = detector.get_confluence_count("AAPL", now)

        assert count == 3

    def test_different_symbols_tracked_separately(self):
        """Test that different symbols are tracked independently."""
        detector = ConfluenceDetector(window_minutes=15)
        now = datetime.now()

        # Record signals for different symbols
        detector.record_signal("AAPL", now - timedelta(minutes=5))
        detector.record_signal("AAPL", now)
        detector.record_signal("TSLA", now)

        # AAPL should have 2 signals (0.10 bonus)
        assert detector.get_bonus("AAPL", now) == 0.10
        # TSLA should have 1 signal (0.0 bonus)
        assert detector.get_bonus("TSLA", now) == 0.0

    def test_custom_bonus_values(self):
        """Test with custom bonus values."""
        detector = ConfluenceDetector(
            window_minutes=15,
            bonus_2_signals=0.15,
            bonus_3_signals=0.25,
        )
        now = datetime.now()

        # Two signals
        detector.record_signal("AAPL", now - timedelta(minutes=5))
        detector.record_signal("AAPL", now)

        assert detector.get_bonus("AAPL", now) == 0.15

        # Add third signal
        detector.record_signal("AAPL", now - timedelta(minutes=2))

        assert detector.get_bonus("AAPL", now) == 0.25

    def test_four_or_more_signals_returns_max_bonus(self):
        """Test that 4+ signals still returns bonus_3_signals (max)."""
        detector = ConfluenceDetector(window_minutes=15)
        now = datetime.now()

        # Four signals within window
        detector.record_signal("AAPL", now - timedelta(minutes=14))
        detector.record_signal("AAPL", now - timedelta(minutes=10))
        detector.record_signal("AAPL", now - timedelta(minutes=5))
        detector.record_signal("AAPL", now)

        bonus = detector.get_bonus("AAPL", now)

        assert bonus == 0.20

    def test_no_signals_returns_zero_bonus(self):
        """Test that no recorded signals returns 0.0 bonus."""
        detector = ConfluenceDetector()
        now = datetime.now()

        bonus = detector.get_bonus("AAPL", now)

        assert bonus == 0.0

    def test_no_signals_returns_zero_count(self):
        """Test that no recorded signals returns 0 count."""
        detector = ConfluenceDetector()
        now = datetime.now()

        count = detector.get_confluence_count("AAPL", now)

        assert count == 0

    def test_custom_window_minutes(self):
        """Test with custom window size."""
        detector = ConfluenceDetector(window_minutes=30)
        now = datetime.now()

        # Signal 25 minutes ago (within 30-minute window)
        detector.record_signal("AAPL", now - timedelta(minutes=25))
        detector.record_signal("AAPL", now)

        # Should count as 2 signals
        bonus = detector.get_bonus("AAPL", now)

        assert bonus == 0.10

    def test_signal_at_exact_window_boundary_included(self):
        """Test that signal exactly at window boundary is included."""
        detector = ConfluenceDetector(window_minutes=15)
        now = datetime.now()

        # Signal exactly 15 minutes ago (at boundary)
        detector.record_signal("AAPL", now - timedelta(minutes=15))
        detector.record_signal("AAPL", now)

        # Should count as 2 signals (boundary is inclusive)
        count = detector.get_confluence_count("AAPL", now)

        assert count == 2
