# src/validation/simulator_engine.py
from dataclasses import dataclass, field
from typing import Any

from src.validation.settings import ValidationSettings


@dataclass
class SimulationResult:
    """Result of running a simulation scenario."""

    passed: bool
    signals_generated: int = 0
    trades_executed: int = 0
    errors: list[str] = field(default_factory=list)


class SimulatorEngine:
    """Orchestrates full pipeline simulation with mock dependencies."""

    def __init__(
        self,
        collector_manager: Any,
        analyzer_manager: Any,
        scoring_engine: Any,
        technical_validator: Any,
        market_gate: Any,
        risk_manager: Any,
        execution_manager: Any,
        journal_manager: Any,
        settings: ValidationSettings | None = None,
    ):
        self.collector_manager = collector_manager
        self.analyzer_manager = analyzer_manager
        self.scoring_engine = scoring_engine
        self.technical_validator = technical_validator
        self.market_gate = market_gate
        self.risk_manager = risk_manager
        self.execution_manager = execution_manager
        self.journal_manager = journal_manager
        self.settings = settings or ValidationSettings()

    async def run_scenario(self, scenario: Any) -> SimulationResult:
        """Run a single scenario and return the result."""
        raise NotImplementedError
