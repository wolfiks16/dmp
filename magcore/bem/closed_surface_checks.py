from __future__ import annotations

from dataclasses import dataclass

from magcore.mesh.adjacency import (
    build_patch_face_to_faces,
    find_patch_boundary_edges,
    find_patch_non_manifold_edges,
    patch_connected_components,
)
from magcore.mesh.surface_mesh import SurfaceMesh


@dataclass(frozen=True, slots=True)
class ClosedSurfaceReport:
    is_closed: bool
    n_faces: int
    n_components: int
    n_boundary_edges: int
    n_non_manifold_edges: int
    boundary_edges: tuple[tuple[int, int], ...]
    non_manifold_edges: dict[tuple[int, int], tuple[int, ...]]
    components: tuple[tuple[int, ...], ...]

    def raise_if_not_closed(self) -> None:
        if not self.is_closed:
            raise ValueError(
                "Face set is not a closed manifold-like surface. "
                f"boundary_edges={self.n_boundary_edges}, "
                f"non_manifold_edges={self.n_non_manifold_edges}"
            )


def check_closed_face_set(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> ClosedSurfaceReport:
    face_indices = tuple(sorted(face_indices))
    boundary_edges = find_patch_boundary_edges(mesh, face_indices)
    non_manifold_edges = find_patch_non_manifold_edges(mesh, face_indices)
    components = patch_connected_components(mesh, face_indices)

    is_closed = (len(boundary_edges) == 0) and (len(non_manifold_edges) == 0)

    return ClosedSurfaceReport(
        is_closed=is_closed,
        n_faces=len(face_indices),
        n_components=len(components),
        n_boundary_edges=len(boundary_edges),
        n_non_manifold_edges=len(non_manifold_edges),
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        components=components,
    )