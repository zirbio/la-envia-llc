# src/validation/scenarios/error_scenarios.py
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.models.social_message import SocialMessage, SourceType
from src.validation.scenarios.base import Scenario

if TYPE_CHECKING:
    from src.validation.simulator_engine import SimulatorEngine


class GateBlocksOutsideHours(Scenario):
    """Market gate should reject signal when market is closed."""

    name = "gate_blocks_outside_hours"

    def __init__(self):
        self.mock_message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="test_gate_1",
            author="unusual_whales",
            content="$TSLA huge call volume!",
            timestamp=datetime.now(timezone.utc),
        )

    async def setup(self) -> None:
        pass

    async def execute(self, engine: "SimulatorEngine") -> None:
        pass

    async def verify(self, engine: "SimulatorEngine") -> bool:
        return True


class RiskLimitBlocksTrade(Scenario):
    """Risk manager should block trade when daily loss limit reached."""

    name = "risk_limit_blocks_trade"

    def __init__(self):
        self.mock_message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="test_risk_1",
            author="unusual_whales",
            content="$NVDA massive sweep!",
            timestamp=datetime.now(timezone.utc),
        )

    async def setup(self) -> None:
        pass

    async def execute(self, engine: "SimulatorEngine") -> None:
        pass

    async def verify(self, engine: "SimulatorEngine") -> bool:
        return True


class CircuitBreakerTriggers(Scenario):
    """Circuit breaker should halt trading after max losses."""

    name = "circuit_breaker_triggers"

    def __init__(self):
        self.mock_message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="test_cb_1",
            author="unusual_whales",
            content="$AMD call sweep detected!",
            timestamp=datetime.now(timezone.utc),
        )

    async def setup(self) -> None:
        pass

    async def execute(self, engine: "SimulatorEngine") -> None:
        pass

    async def verify(self, engine: "SimulatorEngine") -> bool:
        return True


class TechnicalVetoBlocks(Scenario):
    """Technical validator should veto signal when RSI is overbought."""

    name = "technical_veto_blocks"

    def __init__(self):
        self.mock_message = SocialMessage(
            source=SourceType.TWITTER,
            source_id="test_veto_1",
            author="unusual_whales",
            content="$MSFT bullish flow!",
            timestamp=datetime.now(timezone.utc),
        )

    async def setup(self) -> None:
        pass

    async def execute(self, engine: "SimulatorEngine") -> None:
        pass

    async def verify(self, engine: "SimulatorEngine") -> bool:
        return True
