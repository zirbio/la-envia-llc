# tests/validation/test_scenarios.py
import pytest
from src.validation.scenarios.base import Scenario
from src.validation.scenarios.happy_path import BullishSignalExecutesTrade


class TestScenarioBase:
    def test_scenario_has_name(self):
        class TestScenario(Scenario):
            name = "test_scenario"
            async def setup(self): pass
            async def execute(self, engine): pass
            async def verify(self, engine) -> bool: return True

        scenario = TestScenario()
        assert scenario.name == "test_scenario"


class TestHappyPathScenarios:
    def test_bullish_signal_scenario_has_name(self):
        scenario = BullishSignalExecutesTrade()
        assert scenario.name == "bullish_signal_executes_trade"

    def test_bullish_signal_scenario_has_mock_message(self):
        scenario = BullishSignalExecutesTrade()
        assert scenario.mock_message is not None
        assert "$" in scenario.mock_message.content
