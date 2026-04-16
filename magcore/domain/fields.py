from __future__ import annotations

from dataclasses import dataclass
import math

from magcore.typing import Vector3
from magcore.domain.validation import ValidationIssue, error


@dataclass(frozen=True, slots=True)
class ExternalField:
    h_ext: Vector3 = (0.0, 0.0, 0.0)

    def validate_basic(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if len(self.h_ext) != 3 or not all(math.isfinite(x) for x in self.h_ext):
            issues.append(error("external_field.invalid", "External field must be a finite 3D vector.", h_ext=self.h_ext))
        return tuple(issues)