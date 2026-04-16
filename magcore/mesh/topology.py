from __future__ import annotations

from dataclasses import dataclass

from magcore.domain.interfaces import InterfacePatch
from magcore.domain.validation import ValidationIssue, error
from magcore.typing import PatchId, RegionId


@dataclass(frozen=True, slots=True)
class RegionTopology:
    interface_patches: tuple[InterfacePatch, ...]

    def patch_map(self) -> dict[PatchId, InterfacePatch]:
        return {p.patch_id: p for p in self.interface_patches}

    def all_face_indices(self) -> tuple[int, ...]:
        out: list[int] = []
        for p in self.interface_patches:
            out.extend(p.face_indices)
        return tuple(out)

    def region_adjacency(self) -> dict[RegionId, set[RegionId]]:
        adj: dict[RegionId, set[RegionId]] = {}
        for p in self.interface_patches:
            adj.setdefault(p.region_minus_id, set()).add(p.region_plus_id)
            adj.setdefault(p.region_plus_id, set()).add(p.region_minus_id)
        return adj

    def validate_basic(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []

        patch_ids = [p.patch_id for p in self.interface_patches]
        if len(set(patch_ids)) != len(patch_ids):
            issues.append(error("topology.patch_ids.duplicate", "Topology contains duplicate patch IDs."))

        seen_faces: set[int] = set()
        repeated_faces: list[int] = []
        for p in self.interface_patches:
            for f in p.face_indices:
                if f in seen_faces:
                    repeated_faces.append(f)
                seen_faces.add(f)

        if repeated_faces:
            issues.append(error("topology.face.repeated", "A mesh face belongs to more than one patch.", repeated_faces=sorted(set(repeated_faces))))

        return tuple(issues)