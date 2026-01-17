# tests/validation/test_simulator_engine.py
import pytest
from unittest.mock import MagicMock
from src.validation.simulator_engine import SimulatorEngine, SimulationResult
from src.validation.settings import ValidationSettings


class TestSimulatorEngine:
    @pytest.fixture
    def mock_components(self):
        return {
            "collector_manager": MagicMock(),
            "analyzer_manager": MagicMock(),
            "scoring_engine": MagicMock(),
            "technical_validator": MagicMock(),
            "market_gate": MagicMock(),
            "risk_manager": MagicMock(),
            "execution_manager": MagicMock(),
            "journal_manager": MagicMock(),
        }

    def test_init_with_components(self, mock_components):
        engine = SimulatorEngine(**mock_components)
        assert engine.collector_manager == mock_components["collector_manager"]

    def test_init_with_settings(self, mock_components):
        settings = ValidationSettings(scenario_timeout_seconds=60)
        engine = SimulatorEngine(**mock_components, settings=settings)
        assert engine.settings.scenario_timeout_seconds == 60

    def test_simulation_result_dataclass(self):
        result = SimulationResult(
            passed=True,
            signals_generated=5,
            trades_executed=3,
            errors=[],
        )
        assert result.passed is True
        assert result.signals_generated == 5
        assert result.trades_executed == 3

    def test_simulation_result_defaults(self):
        """Verify SimulationResult has correct defaults."""
        result = SimulationResult(passed=False)
        assert result.signals_generated == 0
        assert result.trades_executed == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_run_scenario_not_implemented(self, mock_components):
        """Verify run_scenario raises NotImplementedError."""
        engine = SimulatorEngine(**mock_components)
        with pytest.raises(NotImplementedError):
            await engine.run_scenario(None)
