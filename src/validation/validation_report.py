# src/validation/validation_report.py
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioResult:
    """Result of a single scenario execution."""

    name: str
    passed: bool
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class ValidationReport:
    """Aggregates results from multiple scenarios."""

    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.failed == 0

    def add_result(self, result: ScenarioResult) -> None:
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "all_passed": self.all_passed,
            "results": [
                {"name": r.name, "passed": r.passed, "error": r.error}
                for r in self.results
            ],
        }
