from __future__ import annotations

from dataclasses import dataclass

from magcore.typing import FaceIndex, PatchId, RegionId
from magcore.domain.validation import ValidationIssue, error


@dataclass(frozen=True, slots=True)
class InterfacePatch:
    patch_id: PatchId
    name: str
    region_minus_id: RegionId
    region_plus_id: RegionId
    face_indices: tuple[FaceIndex, ...]

    def validate_basic(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if not self.patch_id.strip():
            issues.append(error("patch.id.empty", "Patch ID must be non-empty."))
        if not self.name.strip():
            issues.append(error("patch.name.empty", "Patch name must be non-empty.", patch_id=self.patch_id))
        if not self.region_minus_id.strip():
            issues.append(error("patch.region_minus.empty", "region_minus_id must be non-empty.", patch_id=self.patch_id))
        if not self.region_plus_id.strip():
            issues.append(error("patch.region_plus.empty", "region_plus_id must be non-empty.", patch_id=self.patch_id))
        if self.region_minus_id == self.region_plus_id and self.region_minus_id.strip():
            issues.append(error("patch.region.same", "Patch must separate two different regions.", patch_id=self.patch_id, region_id=self.region_minus_id))
        if not self.face_indices:
            issues.append(error("patch.faces.empty", "Patch must contain at least one face index.", patch_id=self.patch_id))
        if any((not isinstance(i, int) or i < 0) for i in self.face_indices):
            issues.append(error("patch.faces.invalid", "Patch face_indices must contain non-negative integers.", patch_id=self.patch_id))
        return tuple(issues)