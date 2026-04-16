from __future__ import annotations

from dataclasses import dataclass

from magcore.typing import MaterialId, RegionId
from magcore.domain.validation import ValidationIssue, error


@dataclass(frozen=True, slots=True)
class Region:
    region_id: RegionId
    name: str
    material_id: MaterialId
    is_external: bool = False

    def validate_basic(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if not self.region_id.strip():
            issues.append(error("region.id.empty", "Region ID must be non-empty."))
        if not self.name.strip():
            issues.append(error("region.name.empty", "Region name must be non-empty.", region_id=self.region_id))
        if not self.material_id.strip():
            issues.append(error("region.material_id.empty", "Region material_id must be non-empty.", region_id=self.region_id))
        return tuple(issues)