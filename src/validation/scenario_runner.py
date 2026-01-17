import time
from typing import Any

from src.validation.scenarios.base import Scenario
from src.validation.validation_report import ValidationReport, ScenarioResult
from src.validation.settings import ValidationSettings


class ScenarioRunner:
    """Executes scenarios and collects results."""

    def __init__(
        self,
        engine: Any,
        scenarios: list[Scenario],
        settings: ValidationSettings | None = None,
    ):
        self.engine = engine
        self.scenarios = scenarios
        self.settings = settings or ValidationSettings()

    async def run_all(self) -> ValidationReport:
        """Run all scenarios and return aggregated report."""
        report = ValidationReport()

        for scenario in self.scenarios:
            start = time.perf_counter()
            try:
                await scenario.setup()
                await scenario.execute(self.engine)
                passed = await scenario.verify(self.engine)
                duration_ms = (time.perf_counter() - start) * 1000

                result = ScenarioResult(
                    name=scenario.name,
                    passed=passed,
                    error=None if passed else "Verification failed",
                    duration_ms=duration_ms,
                )
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                result = ScenarioResult(
                    name=scenario.name,
                    passed=False,
                    error=str(e),
                    duration_ms=duration_ms,
                )

            report.add_result(result)

            if not result.passed and self.settings.fail_fast:
                break

        return report
