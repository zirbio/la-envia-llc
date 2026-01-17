# tests/validation/test_validation_report.py
import pytest
from src.validation.validation_report import ValidationReport, ScenarioResult


class TestValidationReport:
    def test_empty_report(self):
        report = ValidationReport()
        assert report.total == 0
        assert report.passed == 0
        assert report.failed == 0

    def test_add_passed_result(self):
        report = ValidationReport()
        result = ScenarioResult(name="test_scenario", passed=True)
        report.add_result(result)
        assert report.total == 1
        assert report.passed == 1
        assert report.failed == 0

    def test_add_failed_result(self):
        report = ValidationReport()
        result = ScenarioResult(name="test_scenario", passed=False, error="assertion failed")
        report.add_result(result)
        assert report.total == 1
        assert report.passed == 0
        assert report.failed == 1

    def test_all_passed_returns_true(self):
        report = ValidationReport()
        report.add_result(ScenarioResult(name="test1", passed=True))
        report.add_result(ScenarioResult(name="test2", passed=True))
        assert report.all_passed is True

    def test_all_passed_returns_false_with_failure(self):
        report = ValidationReport()
        report.add_result(ScenarioResult(name="test1", passed=True))
        report.add_result(ScenarioResult(name="test2", passed=False, error="failed"))
        assert report.all_passed is False

    def test_to_dict(self):
        report = ValidationReport()
        report.add_result(ScenarioResult(name="test1", passed=True))
        d = report.to_dict()
        assert d["total"] == 1
        assert d["passed"] == 1
        assert d["failed"] == 0
        assert len(d["results"]) == 1
