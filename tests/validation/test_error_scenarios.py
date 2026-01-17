# tests/validation/test_error_scenarios.py
import pytest
from src.validation.scenarios.error_scenarios import (
    GateBlocksOutsideHours,
    RiskLimitBlocksTrade,
    CircuitBreakerTriggers,
    TechnicalVetoBlocks,
)


class TestErrorScenarios:
    def test_gate_blocks_outside_hours_has_name(self):
        scenario = GateBlocksOutsideHours()
        assert scenario.name == "gate_blocks_outside_hours"

    def test_risk_limit_blocks_trade_has_name(self):
        scenario = RiskLimitBlocksTrade()
        assert scenario.name == "risk_limit_blocks_trade"

    def test_circuit_breaker_triggers_has_name(self):
        scenario = CircuitBreakerTriggers()
        assert scenario.name == "circuit_breaker_triggers"

    def test_technical_veto_blocks_has_name(self):
        scenario = TechnicalVetoBlocks()
        assert scenario.name == "technical_veto_blocks"

    def test_all_scenarios_have_mock_message(self):
        scenarios = [
            GateBlocksOutsideHours(),
            RiskLimitBlocksTrade(),
            CircuitBreakerTriggers(),
            TechnicalVetoBlocks(),
        ]
        for scenario in scenarios:
            assert scenario.mock_message is not None
