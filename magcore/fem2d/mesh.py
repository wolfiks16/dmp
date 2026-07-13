from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

Edge = tuple[int, int]
Tri = tuple[int, int, int]


def signed_area2(vertices3: np.ndarray) -> float:
    """Удвоенная ЗНАКОВАЯ площадь треугольника (>0 при обходе против часовой стрелки)."""
    p0, p1, p2 = np.asarray(vertices3, dtype=float)
    return float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def triangle_area(vertices3: np.ndarray) -> float:
    return abs(signed_area2(vertices3)) / 2.0


def canonical_edge(edge: tuple[int, int]) -> Edge:
    i, j = int(edge[0]), int(edge[1])
    if i == j:
        raise ValueError("An edge must connect two distinct vertices.")
    return (i, j) if i < j else (j, i)


def canonical_tri(tri: tuple[int, int, int]) -> Tri:
    vals = tuple(sorted(int(i) for i in tri))
    if len(set(vals)) != 3:
        raise ValueError("A triangle must contain exactly three distinct vertices.")
    return vals


def p1_gradients(vertices3: np.ndarray) -> np.ndarray:
    """
    Градиенты линейных базисных функций P1 на треугольнике: (3, 2) массив ∇φ_i (const
    по ячейке). φ_i(p_j)=δ_ij. Для CCW-треугольника с площадью A:
        ∇φ_i = (1/2A)·(y_{i+1}−y_{i+2}, x_{i+2}−x_{i+1})   (циклические индексы).
    Тождества (проверяются тестом): Σ_i ∇φ_i = 0; ∇φ_i·(p_i−p_k)=1 для любого k≠i.
    """
    p = np.asarray(vertices3, dtype=float)
    area2 = signed_area2(p)
    if area2 == 0.0:
        raise ValueError("Degenerate triangle: zero area.")
    grads = np.empty((3, 2), dtype=float)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        grads[i, 0] = (p[j, 1] - p[k, 1]) / area2
        grads[i, 1] = (p[k, 0] - p[j, 0]) / area2
    return grads


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    """
    Конформная треугольная сетка на плоскости (2D-backend). vertices:(N,2), cells:(M,3),
    все треугольники ПОЛОЖИТЕЛЬНО (CCW) ориентированы. 2D-аналог `mesh.mesh.TetraMesh`.
    """

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

        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2).")
        if cells.ndim != 2 or cells.shape[1] != 3:
            raise ValueError("cells must have shape (M, 3).")
        if len(verts) == 0:
            raise ValueError("vertices must be nonempty.")
        if len(cells) == 0:
            raise ValueError("cells must be nonempty.")
        if not np.isfinite(verts).all():
            raise ValueError("vertices must contain only finite values.")
        if np.any(cells < 0) or np.any(cells >= len(verts)):
            raise ValueError("cells contain out-of-range vertex indices.")

        seen: set[Tri] = set()
        for c_idx, cell in enumerate(cells):
            ctuple = tuple(int(v) for v in cell)
            if len(set(ctuple)) != 3:
                raise ValueError(f"Cell {c_idx} has repeated vertex indices.")
            ckey = canonical_tri(ctuple)
            if ckey in seen:
                raise ValueError(f"Duplicate triangle detected at cell {c_idx}.")
            seen.add(ckey)
            if signed_area2(self.cell_vertices(c_idx)) <= 0.0:
                raise ValueError(
                    f"Cell {c_idx} has non-positive signed area. "
                    "All triangles must be consistently CCW-oriented."
                )

        edge_counts = Counter(self.all_edges_with_multiplicity())
        nonmanifold = [e for e, cnt in edge_counts.items() if cnt > 2]
        if nonmanifold:
            raise ValueError(
                "Non-manifold mesh: some edges are shared by more than two triangles."
            )

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    def cell_vertex_indices(self, cell_idx: int) -> Tri:
        c = self.cells[int(cell_idx)]
        return (int(c[0]), int(c[1]), int(c[2]))

    def cell_vertices(self, cell_idx: int) -> np.ndarray:
        return self.vertices[self.cells[int(cell_idx)]]

    def cell_area(self, cell_idx: int) -> float:
        return triangle_area(self.cell_vertices(cell_idx))

    def cell_centroid(self, cell_idx: int) -> np.ndarray:
        return self.cell_vertices(cell_idx).mean(axis=0)

    def cell_edges(self, cell_idx: int) -> tuple[Edge, Edge, Edge]:
        v0, v1, v2 = self.cell_vertex_indices(cell_idx)
        return (
            canonical_edge((v0, v1)),
            canonical_edge((v1, v2)),
            canonical_edge((v2, v0)),
        )

    def all_edges_with_multiplicity(self) -> tuple[Edge, ...]:
        edges: list[Edge] = []
        for c_idx in range(self.n_cells):
            edges.extend(self.cell_edges(c_idx))
        return tuple(edges)

    def all_edges(self) -> tuple[Edge, ...]:
        return tuple(sorted(set(self.all_edges_with_multiplicity())))

    def boundary_edges(self) -> tuple[Edge, ...]:
        counts = Counter(self.all_edges_with_multiplicity())
        return tuple(sorted(e for e, cnt in counts.items() if cnt == 1))

    def boundary_vertices(self) -> tuple[int, ...]:
        verts: set[int] = set()
        for i, j in self.boundary_edges():
            verts.add(i)
            verts.add(j)
        return tuple(sorted(verts))
