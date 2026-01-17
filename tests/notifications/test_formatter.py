# tests/notifications/test_formatter.py
"""Tests for AlertFormatter."""

from datetime import datetime

import pytest


class TestAlertFormatterNewSignal:
    """Tests for format_new_signal."""

    def test_format_long_signal(self):
        """Formats LONG signal correctly."""
        from notifications.alert_formatter import AlertFormatter
        from scoring.models import (
            Direction,
            ScoreComponents,
            ScoreTier,
            TradeRecommendation,
        )

        rec = TradeRecommendation(
            symbol="AAPL",
            direction=Direction.LONG,
            score=85.0,
            tier=ScoreTier.STRONG,
            position_size_percent=5.0,
            entry_price=178.50,
            stop_loss=176.00,
            take_profit=183.00,
            risk_reward_ratio=1.8,
            components=ScoreComponents(
                sentiment_score=90.0,
                technical_score=80.0,
                sentiment_weight=0.4,
                technical_weight=0.35,
                confluence_bonus=0.1,
                credibility_multiplier=1.0,
                time_factor=1.0,
            ),
            reasoning="High sentiment confidence",
            timestamp=datetime(2026, 1, 17, 10, 30),
        )

        formatter = AlertFormatter()
        message = formatter.format_new_signal(rec)

        assert "AAPL" in message
        assert "LONG" in message
        assert "85" in message
        assert "STRONG" in message
        assert "178.50" in message
        assert "176.00" in message
        assert "183.00" in message
        assert "1.8" in message or "1:1.8" in message
        assert "High sentiment confidence" in message

    def test_format_short_signal(self):
        """Formats SHORT signal correctly."""
        from notifications.alert_formatter import AlertFormatter
        from scoring.models import (
            Direction,
            ScoreComponents,
            ScoreTier,
            TradeRecommendation,
        )

        rec = TradeRecommendation(
            symbol="TSLA",
            direction=Direction.SHORT,
            score=72.0,
            tier=ScoreTier.MODERATE,
            position_size_percent=3.0,
            entry_price=250.00,
            stop_loss=255.00,
            take_profit=240.00,
            risk_reward_ratio=2.0,
            components=ScoreComponents(
                sentiment_score=75.0,
                technical_score=70.0,
                sentiment_weight=0.4,
                technical_weight=0.35,
                confluence_bonus=0.05,
                credibility_multiplier=1.0,
                time_factor=0.9,
            ),
            reasoning="Bearish sentiment",
            timestamp=datetime(2026, 1, 17, 11, 0),
        )

        formatter = AlertFormatter()
        message = formatter.format_new_signal(rec)

        assert "TSLA" in message
        assert "SHORT" in message
        assert "72" in message
        assert "MODERATE" in message


class TestAlertFormatterExecution:
    """Tests for format_execution."""

    def test_format_entry_execution(self):
        """Formats entry execution correctly."""
        from execution.models import ExecutionResult
        from notifications.alert_formatter import AlertFormatter

        result = ExecutionResult(
            success=True,
            order_id="order123",
            symbol="AAPL",
            side="buy",
            quantity=50,
            filled_price=178.52,
            error_message=None,
            timestamp=datetime(2026, 1, 17, 10, 35),
        )

        formatter = AlertFormatter()
        message = formatter.format_execution(result, is_entry=True)

        assert "ENTRY" in message
        assert "AAPL" in message
        assert "50" in message
        assert "178.52" in message

    def test_format_exit_execution(self):
        """Formats exit execution correctly."""
        from execution.models import ExecutionResult
        from notifications.alert_formatter import AlertFormatter

        result = ExecutionResult(
            success=True,
            order_id="order456",
            symbol="AAPL",
            side="sell",
            quantity=50,
            filled_price=181.20,
            error_message=None,
            timestamp=datetime(2026, 1, 17, 14, 0),
        )

        formatter = AlertFormatter()
        message = formatter.format_execution(result, is_entry=False)

        assert "EXIT" in message
        assert "AAPL" in message
        assert "50" in message
        assert "181.20" in message


class TestAlertFormatterCircuitBreaker:
    """Tests for format_circuit_breaker."""

    def test_format_circuit_breaker(self):
        """Formats circuit breaker alert correctly."""
        from datetime import date

        from notifications.alert_formatter import AlertFormatter
        from risk.models import DailyRiskState

        state = DailyRiskState(
            date=date(2026, 1, 17),
            realized_pnl=-450.0,
            unrealized_pnl=0.0,
            trades_today=5,
            is_blocked=True,
        )

        formatter = AlertFormatter()
        message = formatter.format_circuit_breaker("Daily loss limit reached", state)

        assert "CIRCUIT BREAKER" in message
        assert "Daily loss limit reached" in message
        assert "450" in message
        assert "5" in message
        assert "blocked" in message.lower()


class TestAlertFormatterDailySummary:
    """Tests for format_daily_summary."""

    def test_format_daily_summary_with_trades(self):
        """Formats daily summary with trades."""
        from notifications.alert_formatter import AlertFormatter

        stats = {
            "date": "2026-01-17",
            "total_trades": 5,
            "winners": 3,
            "losers": 2,
            "gross_pnl": 285.0,
            "largest_win": 180.0,
            "largest_win_symbol": "AAPL",
            "largest_loss": -95.0,
            "largest_loss_symbol": "MSFT",
            "profit_factor": 1.8,
        }

        formatter = AlertFormatter()
        message = formatter.format_daily_summary(stats)

        assert "DAILY SUMMARY" in message
        assert "2026-01-17" in message
        assert "5" in message
        assert "3" in message or "60%" in message
        assert "285" in message
        assert "1.8" in message

    def test_format_daily_summary_no_trades(self):
        """Formats daily summary with no trades."""
        from notifications.alert_formatter import AlertFormatter

        stats = {
            "date": "2026-01-17",
            "total_trades": 0,
            "winners": 0,
            "losers": 0,
            "gross_pnl": 0.0,
        }

        formatter = AlertFormatter()
        message = formatter.format_daily_summary(stats)

        assert "DAILY SUMMARY" in message
        assert "No trades" in message or "0" in message
