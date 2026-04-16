from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import numpy as np

from magcore.femcore.mesh import TetraMesh


Edge = tuple[int, int]


LOCAL_EDGE_VERTEX_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
)


def canonical_edge(i: int, j: int) -> Edge:
    a = int(i)
    b = int(j)
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True, slots=True)
class EdgeTopology:
    global_edges: tuple[Edge, ...]
    cell_to_global_edges: np.ndarray
    cell_edge_signs: np.ndarray

    def __post_init__(self) -> None:
        c2e = np.asarray(self.cell_to_global_edges, dtype=int)
        sgn = np.asarray(self.cell_edge_signs, dtype=int)

        object.__setattr__(self, "cell_to_global_edges", c2e)
        object.__setattr__(self, "cell_edge_signs", sgn)

        if c2e.ndim != 2 or c2e.shape[1] != 6:
            raise ValueError("cell_to_global_edges must have shape (n_cells, 6).")
        if sgn.shape != c2e.shape:
            raise ValueError("cell_edge_signs must have the same shape as cell_to_global_edges.")
        if not np.isin(sgn, (-1, 1)).all():
            raise ValueError("cell_edge_signs must contain only ±1.")

    @property
    def n_edges(self) -> int:
        return len(self.global_edges)


def build_global_edges(mesh: TetraMesh) -> tuple[Edge, ...]:
    edge_set: set[Edge] = set()

    for c_idx in range(mesh.n_cells):
        verts = mesh.cell_vertex_indices(c_idx)
        for i_loc, j_loc in LOCAL_EDGE_VERTEX_PAIRS:
            edge_set.add(canonical_edge(verts[i_loc], verts[j_loc]))

    return tuple(sorted(edge_set))


def build_edge_topology(mesh: TetraMesh) -> EdgeTopology:
    global_edges = build_global_edges(mesh)
    edge_to_idx = {edge: k for k, edge in enumerate(global_edges)}

    c2e = np.zeros((mesh.n_cells, 6), dtype=int)
    sgn = np.zeros((mesh.n_cells, 6), dtype=int)

    for c_idx in range(mesh.n_cells):
        verts = mesh.cell_vertex_indices(c_idx)

        for e_loc, (i_loc, j_loc) in enumerate(LOCAL_EDGE_VERTEX_PAIRS):
            vi = verts[i_loc]
            vj = verts[j_loc]
            gedge = canonical_edge(vi, vj)

            c2e[c_idx, e_loc] = edge_to_idx[gedge]
            sgn[c_idx, e_loc] = +1 if vi < vj else -1

    return EdgeTopology(
        global_edges=global_edges,
        cell_to_global_edges=c2e,
        cell_edge_signs=sgn,
    )


def boundary_edges(mesh: TetraMesh) -> tuple[Edge, ...]:
    """
    Return global edges that belong to at least one boundary face.
    """
    bfaces = mesh.boundary_faces()
    edge_set: set[Edge] = set()

    for i, j, k in bfaces:
        edge_set.add(canonical_edge(i, j))
        edge_set.add(canonical_edge(i, k))
        edge_set.add(canonical_edge(j, k))

    return tuple(sorted(edge_set))


def interior_edges(mesh: TetraMesh) -> tuple[Edge, ...]:
    bnd = set(boundary_edges(mesh))
    all_edges = set(build_global_edges(mesh))
    return tuple(sorted(all_edges.difference(bnd)))