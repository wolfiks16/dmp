from __future__ import annotations

from dataclasses import dataclass

from magcore.domain.fields import ExternalField
from magcore.domain.materials import Material
from magcore.domain.regions import Region
from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.topology import RegionTopology
from magcore.typing import MaterialId, ProblemId, RegionId


@dataclass(frozen=True, slots=True)
class MagnetostaticProblem:
    problem_id: ProblemId
    name: str
    materials: tuple[Material, ...]
    regions: tuple[Region, ...]
    surface_mesh: SurfaceMesh
    topology: RegionTopology
    external_field: ExternalField

    def material_map(self) -> dict[MaterialId, Material]:
        return {m.material_id: m for m in self.materials}

    def region_map(self) -> dict[RegionId, Region]:
        return {r.region_id: r for r in self.regions}

    def get_material(self, material_id: MaterialId) -> Material:
        return self.material_map()[material_id]

    def get_region(self, region_id: RegionId) -> Region:
        return self.region_map()[region_id]

    def external_region(self) -> Region:
        ext = [r for r in self.regions if r.is_external]
        if len(ext) != 1:
            raise ValueError(f"Expected exactly one external region, got {len(ext)}.")
        return ext[0]