"""Tests for risk management data models."""

from datetime import date

import pytest

from risk.models import DailyRiskState, RiskCheckResult


class TestRiskCheckResult:
    """Test suite for RiskCheckResult dataclass."""

    def test_approved_trade(self):
        """Test creating an approved trade result with all fields."""
        result = RiskCheckResult(
            approved=True,
            adjusted_quantity=100,
            adjusted_value=5000.0,
            rejection_reason=None,
            warnings=[],
        )

        assert result.approved is True
        assert result.adjusted_quantity == 100
        assert result.adjusted_value == 5000.0
        assert result.rejection_reason is None
        assert result.warnings == []

    def test_rejected_trade(self):
        """Test creating a rejected trade result with reason."""
        result = RiskCheckResult(
            approved=False,
            adjusted_quantity=0,
            adjusted_value=0.0,
            rejection_reason="Exceeds daily loss limit",
            warnings=[],
        )

        assert result.approved is False
        assert result.adjusted_quantity == 0
        assert result.adjusted_value == 0.0
        assert result.rejection_reason == "Exceeds daily loss limit"
        assert result.warnings == []

    def test_approved_with_warnings(self):
        """Test approved trade with non-blocking warnings."""
        warnings = [
            "Position size approaching 50% of limit",
            "Volatility higher than average",
        ]
        result = RiskCheckResult(
            approved=True,
            adjusted_quantity=50,
            adjusted_value=2500.0,
            rejection_reason=None,
            warnings=warnings,
        )

        assert result.approved is True
        assert result.adjusted_quantity == 50
        assert result.adjusted_value == 2500.0
        assert result.rejection_reason is None
        assert len(result.warnings) == 2
        assert "Position size approaching 50% of limit" in result.warnings
        assert "Volatility higher than average" in result.warnings


class TestDailyRiskState:
    """Test suite for DailyRiskState dataclass."""

    def test_initial_state(self):
        """Test creating a fresh daily state."""
        today = date(2025, 1, 17)
        state = DailyRiskState(
            date=today,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            trades_today=0,
            is_blocked=False,
        )

        assert state.date == today
        assert state.realized_pnl == 0.0
        assert state.unrealized_pnl == 0.0
        assert state.trades_today == 0
        assert state.is_blocked is False

    def test_state_with_losses(self):
        """Test state after losses."""
        today = date(2025, 1, 17)
        state = DailyRiskState(
            date=today,
            realized_pnl=-150.50,
            unrealized_pnl=-75.25,
            trades_today=5,
            is_blocked=False,
        )

        assert state.date == today
        assert state.realized_pnl == -150.50
        assert state.unrealized_pnl == -75.25
        assert state.trades_today == 5
        assert state.is_blocked is False

    def test_blocked_state(self):
        """Test state when daily limit is hit."""
        today = date(2025, 1, 17)
        state = DailyRiskState(
            date=today,
            realized_pnl=-500.0,
            unrealized_pnl=-100.0,
            trades_today=10,
            is_blocked=True,
        )

        assert state.date == today
        assert state.realized_pnl == -500.0
        assert state.unrealized_pnl == -100.0
        assert state.trades_today == 10
        assert state.is_blocked is True
