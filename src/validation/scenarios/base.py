from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.validation.simulator_engine import SimulatorEngine


class Scenario(ABC):
    """Base class for test scenarios."""

    name: str = "unnamed_scenario"

    @abstractmethod
    async def setup(self) -> None:
        """Set up scenario preconditions."""
        pass

    @abstractmethod
    async def execute(self, engine: "SimulatorEngine") -> None:
        """Execute the scenario actions."""
        pass

    @abstractmethod
    async def verify(self, engine: "SimulatorEngine") -> bool:
        """Verify expected outcomes. Returns True if passed."""
        pass
