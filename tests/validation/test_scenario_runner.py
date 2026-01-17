import pytest
from unittest.mock import MagicMock, AsyncMock
from src.validation.scenario_runner import ScenarioRunner
from src.validation.scenarios.base import Scenario
from src.validation.validation_report import ValidationReport
from src.validation.settings import ValidationSettings


class MockScenario(Scenario):
    name = "mock_scenario"

    async def setup(self):
        pass

    async def execute(self, engine):
        pass

    async def verify(self, engine) -> bool:
        return True


class FailingScenario(Scenario):
    name = "failing_scenario"

    async def setup(self):
        pass

    async def execute(self, engine):
        pass

    async def verify(self, engine) -> bool:
        return False


class TestScenarioRunner:
    @pytest.fixture
    def mock_engine(self):
        return MagicMock()

    def test_init_with_scenarios(self, mock_engine):
        scenarios = [MockScenario()]
        runner = ScenarioRunner(engine=mock_engine, scenarios=scenarios)
        assert len(runner.scenarios) == 1

    @pytest.mark.asyncio
    async def test_run_all_returns_report(self, mock_engine):
        scenarios = [MockScenario()]
        runner = ScenarioRunner(engine=mock_engine, scenarios=scenarios)
        report = await runner.run_all()
        assert isinstance(report, ValidationReport)
        assert report.total == 1
        assert report.passed == 1

    @pytest.mark.asyncio
    async def test_run_all_with_failure(self, mock_engine):
        scenarios = [MockScenario(), FailingScenario()]
        runner = ScenarioRunner(engine=mock_engine, scenarios=scenarios)
        report = await runner.run_all()
        assert report.total == 2
        assert report.passed == 1
        assert report.failed == 1

    @pytest.mark.asyncio
    async def test_fail_fast_stops_on_first_failure(self, mock_engine):
        settings = ValidationSettings(fail_fast=True)
        scenarios = [FailingScenario(), MockScenario()]
        runner = ScenarioRunner(engine=mock_engine, scenarios=scenarios, settings=settings)
        report = await runner.run_all()
        assert report.total == 1  # Stopped after first failure
        assert report.failed == 1
