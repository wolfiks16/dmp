from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import numpy as np


Edge = tuple[int, int]
Face = tuple[int, int, int]
Cell = tuple[int, int, int, int]


def oriented_tetra_volume6(vertices4: np.ndarray) -> float:
    v0, v1, v2, v3 = np.asarray(vertices4, dtype=float)
    J = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
    return float(np.linalg.det(J))


def tetra_volume(vertices4: np.ndarray) -> float:
    return abs(oriented_tetra_volume6(vertices4)) / 6.0


def canonical_edge(edge: tuple[int, int]) -> Edge:
    i, j = (int(edge[0]), int(edge[1]))
    if i == j:
        raise ValueError("An edge must connect two distinct vertices.")
    return (i, j) if i < j else (j, i)


def canonical_face(face: tuple[int, int, int]) -> Face:
    vals = tuple(sorted(int(i) for i in face))
    if len(set(vals)) != 3:
        raise ValueError("A face must contain exactly three distinct vertices.")
    return vals


def canonical_cell(cell: tuple[int, int, int, int]) -> Cell:
    vals = tuple(sorted(int(i) for i in cell))
    if len(set(vals)) != 4:
        raise ValueError("A cell must contain exactly four distinct vertices.")
    return vals


@dataclass(frozen=True, slots=True)
class TetraMesh:
    vertices: np.ndarray
    cells: np.ndarray

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=float)
        cells = np.asarray(self.cells, dtype=int)

        object.__setattr__(self, "vertices", verts)
        object.__setattr__(self, "cells", cells)

        self.validate()

    def validate(self) -> None:
        verts = self.vertices
        cells = self.cells

        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3).")
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("cells must have shape (M, 4).")
        if len(verts) == 0:
            raise ValueError("vertices must be nonempty.")
        if len(cells) == 0:
            raise ValueError("cells must be nonempty.")
        if not np.isfinite(verts).all():
            raise ValueError("vertices must contain only finite values.")
        if np.any(cells < 0) or np.any(cells >= len(verts)):
            raise ValueError("cells contain out-of-range vertex indices.")

        seen_cells: set[Cell] = set()

        for c_idx, cell in enumerate(cells):
            ctuple = tuple(int(v) for v in cell)

            if len(set(ctuple)) != 4:
                raise ValueError(f"Cell {c_idx} has repeated vertex indices.")

            ckey = canonical_cell(ctuple)
            if ckey in seen_cells:
                raise ValueError(f"Duplicate tetrahedron detected at cell {c_idx}.")
            seen_cells.add(ckey)

            vol6 = oriented_tetra_volume6(self.cell_vertices(c_idx))
            if vol6 <= 0.0:
                raise ValueError(
                    f"Cell {c_idx} has non-positive oriented volume. "
                    "All tetrahedra must be consistently positively oriented."
                )

        face_counts = Counter(self.all_faces())
        nonmanifold_faces = [face for face, cnt in face_counts.items() if cnt > 2]
        if nonmanifold_faces:
            raise ValueError(
                "Non-manifold mesh: some faces are shared by more than two tetrahedra."
            )

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    def cell_vertex_indices(self, cell_idx: int) -> Cell:
        cell = self.cells[int(cell_idx)]
        return (int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3]))

    def cell_vertices(self, cell_idx: int) -> np.ndarray:
        return self.vertices[self.cells[int(cell_idx)]]

    def cell_volume(self, cell_idx: int) -> float:
        return tetra_volume(self.cell_vertices(cell_idx))

    def cell_centroid(self, cell_idx: int) -> np.ndarray:
        return self.cell_vertices(cell_idx).mean(axis=0)

    def cell_edges(self, cell_idx: int) -> tuple[Edge, Edge, Edge, Edge, Edge, Edge]:
        v0, v1, v2, v3 = self.cell_vertex_indices(cell_idx)
        return (
            canonical_edge((v0, v1)),
            canonical_edge((v0, v2)),
            canonical_edge((v0, v3)),
            canonical_edge((v1, v2)),
            canonical_edge((v1, v3)),
            canonical_edge((v2, v3)),
        )

    def cell_faces(self, cell_idx: int) -> tuple[Face, Face, Face, Face]:
        v0, v1, v2, v3 = self.cell_vertex_indices(cell_idx)
        return (
            canonical_face((v1, v2, v3)),
            canonical_face((v0, v3, v2)),
            canonical_face((v0, v1, v3)),
            canonical_face((v0, v2, v1)),
        )

    def all_edges(self) -> tuple[Edge, ...]:
        edges: set[Edge] = set()
        for c_idx in range(self.n_cells):
            edges.update(self.cell_edges(c_idx))
        return tuple(sorted(edges))

    def edge_count(self) -> int:
        return len(self.all_edges())

    def all_faces(self) -> tuple[Face, ...]:
        faces: list[Face] = []
        for c_idx in range(self.n_cells):
            faces.extend(self.cell_faces(c_idx))
        return tuple(faces)

    def boundary_faces(self) -> tuple[Face, ...]:
        counts = Counter(self.all_faces())
        bfaces = [face for face, cnt in counts.items() if cnt == 1]
        return tuple(sorted(bfaces))

    def boundary_face_count(self) -> int:
        return len(self.boundary_faces())

    def boundary_edges(self) -> tuple[Edge, ...]:
        edges: set[Edge] = set()
        for face in self.boundary_faces():
            i, j, k = face
            edges.add(canonical_edge((i, j)))
            edges.add(canonical_edge((j, k)))
            edges.add(canonical_edge((k, i)))
        return tuple(sorted(edges))

    def boundary_vertices(self) -> tuple[int, ...]:
        verts: set[int] = set()
        for face in self.boundary_faces():
            verts.update(face)
        return tuple(sorted(verts))