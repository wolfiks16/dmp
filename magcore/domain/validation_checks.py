from __future__ import annotations

from collections import Counter

from magcore.domain.interfaces import InterfacePatch
from magcore.enums import MaterialKind
from magcore.domain.materials import Material, PermanentMagnetMaterial
from magcore.domain.problem import MagnetostaticProblem
from magcore.domain.regions import Region
from magcore.domain.validation import ValidationIssue, ValidationReport, error, warning
from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.topology import RegionTopology
from magcore.mesh.adjacency import (
    find_boundary_edges,
    find_non_manifold_edges,
    find_patch_boundary_edges,
    find_patch_non_manifold_edges,
    patch_connected_components,
)
from magcore.mesh.normals import (
    find_orientation_conflicts,
    orientability_check,
)
from magcore.mesh.quality import (
    mesh_quality_summary,
    find_tiny_edges,
)


def validate_materials(materials: tuple[Material, ...]) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    issues.extend(i for m in materials for i in m.validate())

    ids = [m.material_id for m in materials]
    duplicates = [x for x, c in Counter(ids).items() if c > 1]
    if duplicates:
        issues.append(error("materials.ids.duplicate", "Duplicate material IDs detected.", duplicates=duplicates))

    return tuple(issues)


def validate_regions(regions: tuple[Region, ...], materials: tuple[Material, ...]) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    issues.extend(i for r in regions for i in r.validate_basic())

    region_ids = [r.region_id for r in regions]
    duplicates = [x for x, c in Counter(region_ids).items() if c > 1]
    if duplicates:
        issues.append(error("regions.ids.duplicate", "Duplicate region IDs detected.", duplicates=duplicates))

    material_ids = {m.material_id for m in materials}
    for r in regions:
        if r.material_id not in material_ids:
            issues.append(error("regions.material.missing", "Region references a missing material.", region_id=r.region_id, material_id=r.material_id))

    ext_regions = [r for r in regions if r.is_external]
    if len(ext_regions) != 1:
        issues.append(error("regions.external.count", "Exactly one external region is required.", count=len(ext_regions)))

    return tuple(issues)


def validate_surface_mesh(mesh: SurfaceMesh) -> tuple[ValidationIssue, ...]:
    return mesh.validate_basic()


def validate_topology(topology: RegionTopology, mesh: SurfaceMesh, regions: tuple[Region, ...]) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    issues.extend(topology.validate_basic())

    region_ids = {r.region_id for r in regions}
    all_faces = set(range(mesh.n_faces))
    topology_faces = set(topology.all_face_indices())

    for p in topology.interface_patches:
        issues.extend(p.validate_basic())
        if p.region_minus_id not in region_ids:
            issues.append(error("topology.patch.region_minus.missing", "Patch references missing region_minus_id.", patch_id=p.patch_id, region_id=p.region_minus_id))
        if p.region_plus_id not in region_ids:
            issues.append(error("topology.patch.region_plus.missing", "Patch references missing region_plus_id.", patch_id=p.patch_id, region_id=p.region_plus_id))
        for f in p.face_indices:
            if f >= mesh.n_faces:
                issues.append(error("topology.patch.face.out_of_range", "Patch references face index outside mesh range.", patch_id=p.patch_id, face_idx=f, n_faces=mesh.n_faces))

    if topology_faces != all_faces:
        missing = sorted(all_faces - topology_faces)
        extra = sorted(topology_faces - all_faces)
        if missing:
            issues.append(error("topology.faces.uncovered", "Some mesh faces are not assigned to any patch.", missing_faces=missing[:50], missing_count=len(missing)))
        if extra:
            issues.append(error("topology.faces.extra", "Topology contains non-existent face indices.", extra_faces=extra[:50], extra_count=len(extra)))

    return tuple(issues)


def validate_problem(problem: MagnetostaticProblem) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if not problem.problem_id.strip():
        issues.append(error("problem.id.empty", "Problem ID must be non-empty."))
    if not problem.name.strip():
        issues.append(error("problem.name.empty", "Problem name must be non-empty."))

    issues.extend(validate_materials(problem.materials))
    issues.extend(validate_regions(problem.regions, problem.materials))
    issues.extend(validate_surface_mesh(problem.surface_mesh))
    issues.extend(validate_topology(problem.topology, problem.surface_mesh, problem.regions))
    issues.extend(problem.external_field.validate_basic())
    issues.extend(validate_topology_geometry(problem.surface_mesh, problem.topology))

    # Cross-check external region material
    material_map = {m.material_id: m for m in problem.materials}
    external_regions = [r for r in problem.regions if r.is_external]
    if len(external_regions) == 1:
        ext_region = external_regions[0]
        ext_material = material_map.get(ext_region.material_id)
        if isinstance(ext_material, PermanentMagnetMaterial):
            issues.append(error("problem.external_region.permanent_magnet", "External region cannot use a permanent magnet material.", region_id=ext_region.region_id, material_id=ext_region.material_id))
        elif ext_material is not None and ext_material.kind != MaterialKind.AIR:
            issues.append(warning("problem.external_region.not_air", "External region is expected to use an air-like material.", region_id=ext_region.region_id, material_kind=ext_material.kind.value))

    return ValidationReport(tuple(issues))

def validate_mesh_topology(mesh: SurfaceMesh) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []

    non_manifold = find_non_manifold_edges(mesh)
    if non_manifold:
        issues.append(error(
            "mesh.edge.non_manifold",
            "Mesh contains non-manifold edges.",
            count=len(non_manifold),
            sample=list(non_manifold.items())[:10],
        ))

    boundary_edges = find_boundary_edges(mesh)
    if boundary_edges:
        issues.append(warning(
            "mesh.edge.boundary.present",
            "Mesh contains boundary edges.",
            count=len(boundary_edges),
            sample=boundary_edges[:10],
        ))

    return tuple(issues)


def validate_patch_geometry(mesh: SurfaceMesh, patch: InterfacePatch) -> tuple[ValidationIssue, ...]:
    if any(f < 0 or f >= mesh.n_faces for f in patch.face_indices):
        return ()
    
    issues: list[ValidationIssue] = []

    components = patch_connected_components(mesh, patch.face_indices)
    if len(components) > 1:
        issues.append(warning(
            "patch.faces.disconnected",
            "Patch consists of multiple disconnected face components.",
            patch_id=patch.patch_id,
            n_components=len(components),
            components=components[:10],
        ))

    non_manifold = find_patch_non_manifold_edges(mesh, patch.face_indices)
    if non_manifold:
        issues.append(error(
            "patch.edge.non_manifold",
            "Patch contains non-manifold edges.",
            patch_id=patch.patch_id,
            count=len(non_manifold),
            sample=list(non_manifold.items())[:10],
        ))

    conflicts = find_orientation_conflicts(mesh, patch.face_indices)
    if conflicts:
        issues.append(error(
            "patch.orientation.conflict",
            "Patch contains neighboring faces with inconsistent winding.",
            patch_id=patch.patch_id,
            count=len(conflicts),
            sample=conflicts[:10],
        ))

    if not orientability_check(mesh, patch.face_indices):
        issues.append(error(
            "patch.orientation.non_orientable",
            "Patch is not orientable under local adjacency constraints.",
            patch_id=patch.patch_id,
        ))

    summary = mesh_quality_summary(mesh, patch.face_indices)
    tiny_edges = find_tiny_edges(mesh, edge_threshold=1.0e-12, face_indices=patch.face_indices)
    if tiny_edges:
        issues.append(warning(
            "patch.quality.tiny_edges",
            "Patch contains very small edges.",
            patch_id=patch.patch_id,
            count=len(tiny_edges),
            sample=tiny_edges[:10],
        ))

    if summary["max_aspect_ratio"] > 1.0e3:
        issues.append(warning(
            "patch.quality.bad_aspect_ratio",
            "Patch contains highly stretched triangles.",
            patch_id=patch.patch_id,
            max_aspect_ratio=summary["max_aspect_ratio"],
        ))

    return tuple(issues)


def validate_topology_geometry(mesh: SurfaceMesh, topology: RegionTopology) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    issues.extend(validate_mesh_topology(mesh))
    for patch in topology.interface_patches:
        issues.extend(validate_patch_geometry(mesh, patch))
    return tuple(issues)