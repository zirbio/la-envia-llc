"""Tests for gate data models."""

from datetime import datetime

import pytest

from gate.models import GateCheckResult, GateStatus


class TestGateCheckResult:
    """Tests for GateCheckResult dataclass."""

    def test_create_passed_check(self) -> None:
        """GateCheckResult can be created with passed=True."""
        result = GateCheckResult(
            name="trading_hours",
            passed=True,
            reason=None,
            data={"current_time": "10:30"},
        )
        assert result.name == "trading_hours"
        assert result.passed is True
        assert result.reason is None
        assert result.data == {"current_time": "10:30"}

    def test_create_failed_check(self) -> None:
        """GateCheckResult can be created with passed=False and reason."""
        result = GateCheckResult(
            name="vix",
            passed=False,
            reason="VIX above maximum threshold",
            data={"vix_value": 35.5},
        )
        assert result.name == "vix"
        assert result.passed is False
        assert result.reason == "VIX above maximum threshold"
        assert result.data["vix_value"] == 35.5


class TestGateStatus:
    """Tests for GateStatus dataclass."""

    def test_create_open_status(self) -> None:
        """GateStatus can be created with is_open=True."""
        now = datetime.now()
        checks = [
            GateCheckResult(name="hours", passed=True, reason=None, data={}),
            GateCheckResult(name="volume", passed=True, reason=None, data={}),
        ]
        status = GateStatus(
            timestamp=now,
            is_open=True,
            checks=checks,
            position_size_factor=1.0,
        )
        assert status.is_open is True
        assert status.position_size_factor == 1.0
        assert len(status.checks) == 2

    def test_create_closed_status(self) -> None:
        """GateStatus can be created with is_open=False."""
        now = datetime.now()
        checks = [
            GateCheckResult(name="hours", passed=False, reason="Market closed", data={}),
        ]
        status = GateStatus(
            timestamp=now,
            is_open=False,
            checks=checks,
            position_size_factor=0.0,
        )
        assert status.is_open is False
        assert status.position_size_factor == 0.0

    def test_get_failed_checks_returns_only_failed(self) -> None:
        """get_failed_checks returns only checks that did not pass."""
        now = datetime.now()
        checks = [
            GateCheckResult(name="hours", passed=True, reason=None, data={}),
            GateCheckResult(name="vix", passed=False, reason="VIX too high", data={}),
            GateCheckResult(name="volume", passed=True, reason=None, data={}),
        ]
        status = GateStatus(
            timestamp=now,
            is_open=False,
            checks=checks,
            position_size_factor=0.0,
        )
        failed = status.get_failed_checks()
        assert len(failed) == 1
        assert failed[0].name == "vix"

    def test_get_failed_checks_returns_empty_when_all_pass(self) -> None:
        """get_failed_checks returns empty list when all checks pass."""
        now = datetime.now()
        checks = [
            GateCheckResult(name="hours", passed=True, reason=None, data={}),
            GateCheckResult(name="volume", passed=True, reason=None, data={}),
        ]
        status = GateStatus(
            timestamp=now,
            is_open=True,
            checks=checks,
            position_size_factor=1.0,
        )
        failed = status.get_failed_checks()
        assert len(failed) == 0
