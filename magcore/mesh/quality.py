from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.adjacency import build_patch_edge_to_faces


def face_edge_lengths(mesh: SurfaceMesh, face_idx: int) -> tuple[float, float, float]:
    tri = mesh.face_vertices(face_idx)
    l01 = float(np.linalg.norm(tri[1] - tri[0]))
    l12 = float(np.linalg.norm(tri[2] - tri[1]))
    l20 = float(np.linalg.norm(tri[0] - tri[2]))
    return (l01, l12, l20)


def face_aspect_ratio(mesh: SurfaceMesh, face_idx: int) -> float:
    lengths = face_edge_lengths(mesh, face_idx)
    mn = min(lengths)
    mx = max(lengths)
    if mn == 0.0:
        return float("inf")
    return mx / mn


def mesh_quality_summary(mesh: SurfaceMesh, face_indices: tuple[int, ...] | None = None) -> dict[str, float | int]:
    if face_indices is None:
        face_indices = tuple(range(mesh.n_faces))

    if not face_indices:
        return {
            "n_faces": 0,
            "min_area": 0.0,
            "max_area": 0.0,
            "mean_area": 0.0,
            "min_edge": 0.0,
            "max_edge": 0.0,
            "max_aspect_ratio": 0.0,
        }

    areas = np.array([mesh.face_area(f) for f in face_indices], dtype=float)
    aspect = np.array([face_aspect_ratio(mesh, f) for f in face_indices], dtype=float)

    all_edges: list[float] = []
    for f in face_indices:
        all_edges.extend(face_edge_lengths(mesh, f))
    edge_arr = np.array(all_edges, dtype=float)

    return {
        "n_faces": int(len(face_indices)),
        "min_area": float(np.min(areas)),
        "max_area": float(np.max(areas)),
        "mean_area": float(np.mean(areas)),
        "min_edge": float(np.min(edge_arr)),
        "max_edge": float(np.max(edge_arr)),
        "max_aspect_ratio": float(np.max(aspect)),
    }


def find_near_degenerate_faces(mesh: SurfaceMesh, area_threshold: float, face_indices: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if face_indices is None:
        face_indices = tuple(range(mesh.n_faces))
    return tuple(f for f in face_indices if mesh.face_area(f) <= area_threshold)


def find_tiny_edges(mesh: SurfaceMesh, edge_threshold: float, face_indices: tuple[int, ...] | None = None) -> tuple[tuple[int, int], ...]:
    if face_indices is None:
        face_indices = tuple(range(mesh.n_faces))

    tiny: set[tuple[int, int]] = set()
    for f in face_indices:
        face = mesh.faces[f]
        pairs = ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0])))
        tri = mesh.face_vertices(f)
        lengths = (
            float(np.linalg.norm(tri[1] - tri[0])),
            float(np.linalg.norm(tri[2] - tri[1])),
            float(np.linalg.norm(tri[0] - tri[2])),
        )
        for (a, b), l in zip(pairs, lengths):
            if l <= edge_threshold:
                tiny.add((a, b) if a < b else (b, a))
    return tuple(sorted(tiny))