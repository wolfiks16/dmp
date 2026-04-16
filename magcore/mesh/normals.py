from __future__ import annotations

from collections import deque

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.adjacency import build_patch_face_to_faces


def face_oriented_edges(face: np.ndarray | tuple[int, int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
    return ((v0, v1), (v1, v2), (v2, v0))


def shared_edge_orientation(face_a: np.ndarray | tuple[int, int, int], face_b: np.ndarray | tuple[int, int, int]) -> int | None:
    edges_a = face_oriented_edges(face_a)
    edges_b = face_oriented_edges(face_b)

    for ea in edges_a:
        a_set = {ea[0], ea[1]}
        for eb in edges_b:
            if a_set == {eb[0], eb[1]}:
                return -1 if ea == (eb[1], eb[0]) else +1
    return None


def compute_face_normals(mesh: SurfaceMesh, face_indices: tuple[int, ...] | None = None) -> dict[int, np.ndarray]:
    if face_indices is None:
        face_indices = tuple(range(mesh.n_faces))
    return {f: mesh.face_normal(f) for f in face_indices}


def find_orientation_conflicts(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    face_to_faces = build_patch_face_to_faces(mesh, face_indices)
    conflicts: set[tuple[int, int]] = set()

    for f in face_indices:
        for nbr in face_to_faces.get(f, ()):
            if f < nbr:
                sign = shared_edge_orientation(mesh.faces[f], mesh.faces[nbr])
                if sign == +1:
                    conflicts.add((f, nbr))

    return tuple(sorted(conflicts))


def estimate_patch_reference_normal(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> np.ndarray | None:
    acc = np.zeros(3, dtype=float)
    for f in face_indices:
        area = mesh.face_area(f)
        normal = mesh.face_normal(f)
        acc += area * normal

    norm = np.linalg.norm(acc)
    if norm == 0.0:
        return None
    return acc / norm


def find_strong_normal_flips(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    threshold_cos: float = -0.5,
) -> tuple[tuple[int, int], ...]:
    face_to_faces = build_patch_face_to_faces(mesh, face_indices)
    normals = compute_face_normals(mesh, face_indices)
    flips: set[tuple[int, int]] = set()

    for f in face_indices:
        for nbr in face_to_faces.get(f, ()):
            if f < nbr:
                c = float(np.dot(normals[f], normals[nbr]))
                if c < threshold_cos:
                    flips.add((f, nbr))

    return tuple(sorted(flips))


def orientability_check(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> bool:
    face_set = set(face_indices)
    face_to_faces = build_patch_face_to_faces(mesh, face_indices)
    assigned: dict[int, int] = {}

    for start in face_indices:
        if start in assigned:
            continue

        assigned[start] = +1
        q = deque([start])

        while q:
            cur = q.popleft()
            cur_face = mesh.faces[cur]

            for nbr in face_to_faces.get(cur, ()):
                sign = shared_edge_orientation(cur_face, mesh.faces[nbr])
                if sign is None:
                    continue

                required = -1 if sign == +1 else +1
                candidate = assigned[cur] * required

                if nbr not in assigned:
                    assigned[nbr] = candidate
                    q.append(nbr)
                elif assigned[nbr] != candidate:
                    return False

    return True