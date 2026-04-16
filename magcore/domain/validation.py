from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from magcore.enums import ValidationLevel
from magcore.exceptions import DomainValidationError


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    level: ValidationLevel
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    def has_errors(self) -> bool:
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        return any(issue.level == ValidationLevel.WARNING for issue in self.issues)

    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.level == ValidationLevel.ERROR)

    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.level == ValidationLevel.WARNING)

    def raise_if_errors(self) -> None:
        if self.has_errors():
            lines = [f"[{i.code}] {i.message}" for i in self.errors()]
            raise DomainValidationError("\n".join(lines))

    def extend(self, more: Iterable[ValidationIssue]) -> "ValidationReport":
        return ValidationReport(self.issues + tuple(more))


def error(code: str, message: str, **context: Any) -> ValidationIssue:
    return ValidationIssue(
        level=ValidationLevel.ERROR,
        code=code,
        message=message,
        context=context,
    )


def warning(code: str, message: str, **context: Any) -> ValidationIssue:
    return ValidationIssue(
        level=ValidationLevel.WARNING,
        code=code,
        message=message,
        context=context,
    )