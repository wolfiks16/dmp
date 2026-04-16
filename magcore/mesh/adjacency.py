from __future__ import annotations

from collections import defaultdict, deque
from typing import TypeAlias

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh

Edge: TypeAlias = tuple[int, int]


def canonical_edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def face_edges(face: np.ndarray | tuple[int, int, int]) -> tuple[Edge, Edge, Edge]:
    v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
    return (
        canonical_edge(v0, v1),
        canonical_edge(v1, v2),
        canonical_edge(v2, v0),
    )


def build_vertex_to_faces(mesh: SurfaceMesh) -> dict[int, tuple[int, ...]]:
    out: dict[int, list[int]] = defaultdict(list)
    for f_idx, face in enumerate(mesh.faces):
        for v in face:
            out[int(v)].append(f_idx)
    return {k: tuple(v) for k, v in out.items()}


def build_edge_to_faces(mesh: SurfaceMesh) -> dict[Edge, tuple[int, ...]]:
    out: dict[Edge, list[int]] = defaultdict(list)
    for f_idx, face in enumerate(mesh.faces):
        for e in face_edges(face):
            out[e].append(f_idx)
    return {k: tuple(v) for k, v in out.items()}


def build_face_to_faces(mesh: SurfaceMesh) -> dict[int, tuple[int, ...]]:
    edge_to_faces = build_edge_to_faces(mesh)
    nbrs: dict[int, set[int]] = {i: set() for i in range(mesh.n_faces)}
    for faces in edge_to_faces.values():
        if len(faces) >= 2:
            for i in faces:
                for j in faces:
                    if i != j:
                        nbrs[i].add(j)
    return {k: tuple(sorted(v)) for k, v in nbrs.items()}


def find_boundary_edges(mesh: SurfaceMesh) -> tuple[Edge, ...]:
    edge_to_faces = build_edge_to_faces(mesh)
    return tuple(sorted(e for e, faces in edge_to_faces.items() if len(faces) == 1))


def find_non_manifold_edges(mesh: SurfaceMesh) -> dict[Edge, tuple[int, ...]]:
    edge_to_faces = build_edge_to_faces(mesh)
    return {e: faces for e, faces in edge_to_faces.items() if len(faces) > 2}


def build_patch_edge_to_faces(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> dict[Edge, tuple[int, ...]]:
    face_set = set(face_indices)
    out: dict[Edge, list[int]] = defaultdict(list)
    for f_idx in face_indices:
        face = mesh.faces[f_idx]
        for e in face_edges(face):
            out[e].append(f_idx)
    return {k: tuple(v) for k, v in out.items()}


def build_patch_face_to_faces(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
    edge_to_faces = build_patch_edge_to_faces(mesh, face_indices)
    nbrs: dict[int, set[int]] = {f: set() for f in face_indices}
    for faces in edge_to_faces.values():
        if len(faces) >= 2:
            for i in faces:
                for j in faces:
                    if i != j:
                        nbrs[i].add(j)
    return {k: tuple(sorted(v)) for k, v in nbrs.items()}


def find_patch_boundary_edges(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> tuple[Edge, ...]:
    edge_to_faces = build_patch_edge_to_faces(mesh, face_indices)
    return tuple(sorted(e for e, faces in edge_to_faces.items() if len(faces) == 1))


def find_patch_non_manifold_edges(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> dict[Edge, tuple[int, ...]]:
    edge_to_faces = build_patch_edge_to_faces(mesh, face_indices)
    return {e: faces for e, faces in edge_to_faces.items() if len(faces) > 2}


def patch_connected_components(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    face_to_faces = build_patch_face_to_faces(mesh, face_indices)
    unvisited = set(face_indices)
    comps: list[tuple[int, ...]] = []

    while unvisited:
        start = next(iter(unvisited))
        q = deque([start])
        comp: list[int] = []
        unvisited.remove(start)

        while q:
            cur = q.popleft()
            comp.append(cur)
            for nbr in face_to_faces.get(cur, ()):
                if nbr in unvisited:
                    unvisited.remove(nbr)
                    q.append(nbr)

        comps.append(tuple(sorted(comp)))

    return tuple(comps)