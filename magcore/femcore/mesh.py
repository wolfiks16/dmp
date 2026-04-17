from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import numpy as np


Face = tuple[int, int, int]
Cell = tuple[int, int, int, int]


def oriented_tetra_volume6(vertices4: np.ndarray) -> float:
    v0, v1, v2, v3 = np.asarray(vertices4, dtype=float)
    J = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
    return float(np.linalg.det(J))


def tetra_volume(vertices4: np.ndarray) -> float:
    return abs(oriented_tetra_volume6(vertices4)) / 6.0


def canonical_face(face: tuple[int, int, int]) -> Face:
    return tuple(sorted(int(i) for i in face))


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

        for c_idx, cell in enumerate(cells):
            if len(set(int(v) for v in cell)) != 4:
                raise ValueError(f"Cell {c_idx} has repeated vertex indices.")
            vol6 = oriented_tetra_volume6(self.cell_vertices(c_idx))
            if vol6 <= 0.0:
                raise ValueError(
                    f"Cell {c_idx} has non-positive oriented volume. "
                    "All tetrahedra must be consistently positively oriented."
                )

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    def cell_vertex_indices(self, cell_idx: int) -> tuple[int, int, int, int]:
        cell = self.cells[int(cell_idx)]
        return (int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3]))

    def cell_vertices(self, cell_idx: int) -> np.ndarray:
        return self.vertices[self.cells[int(cell_idx)]]

    def cell_volume(self, cell_idx: int) -> float:
        return tetra_volume(self.cell_vertices(cell_idx))

    def cell_centroid(self, cell_idx: int) -> np.ndarray:
        return self.cell_vertices(cell_idx).mean(axis=0)

    def cell_faces(self, cell_idx: int) -> tuple[Face, Face, Face, Face]:
        v0, v1, v2, v3 = self.cell_vertex_indices(cell_idx)
        return (
            canonical_face((v1, v2, v3)),
            canonical_face((v0, v3, v2)),
            canonical_face((v0, v1, v3)),
            canonical_face((v0, v2, v1)),
        )

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
