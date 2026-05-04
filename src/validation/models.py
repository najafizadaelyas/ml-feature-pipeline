"""Dataclasses for validation results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnValidationResult:
    column: str
    check: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationResult:
    passed: bool
    results: list[ColumnValidationResult] = field(default_factory=list)

    def failures(self) -> list[ColumnValidationResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        total = len(self.results)
        failed = len(self.failures())
        return f"{total - failed}/{total} checks passed" + (
            f"; failures: {[f.detail for f in self.failures()]}" if failed else ""
        )
