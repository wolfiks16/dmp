from __future__ import annotations

import numpy as np

from magcore.femcore.edge_topology import (
    LOCAL_EDGE_VERTEX_PAIRS,
    boundary_edges,
    build_edge_topology,
    build_global_edges,
    interior_edges,
)
from magcore.mesh.mesh import TetraMesh


def make_single_tetra_mesh() -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=int)
    return TetraMesh(vertices=vertices, cells=cells)


def make_two_tetra_mesh() -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # 0
            [1.0, 0.0, 0.0],   # 1
            [0.0, 1.0, 0.0],   # 2
            [0.0, 0.0, 1.0],   # 3
            [0.0, 0.0, -1.0],  # 4
        ],
        dtype=float,
    )
    cells = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 4],
        ],
        dtype=int,
    )
    return TetraMesh(vertices=vertices, cells=cells)


def test_local_edge_vertex_pairs_has_six_edges():
    assert len(LOCAL_EDGE_VERTEX_PAIRS) == 6


def test_build_global_edges_single_tetra():
    mesh = make_single_tetra_mesh()
    edges = build_global_edges(mesh)

    assert len(edges) == 6
    assert (0, 1) in edges
    assert (0, 2) in edges
    assert (0, 3) in edges
    assert (1, 2) in edges
    assert (1, 3) in edges
    assert (2, 3) in edges


def test_build_global_edges_two_tetra_shared_face_not_duplicated():
    mesh = make_two_tetra_mesh()
    edges = build_global_edges(mesh)

    assert len(edges) == 9


def test_build_edge_topology_shapes_single_tetra():
    mesh = make_single_tetra_mesh()
    topo = build_edge_topology(mesh)

    assert topo.n_edges == 6
    assert topo.cell_to_global_edges.shape == (1, 6)
    assert topo.cell_edge_signs.shape == (1, 6)


def test_build_edge_topology_signs_single_tetra_all_positive():
    mesh = make_single_tetra_mesh()
    topo = build_edge_topology(mesh)

    assert np.all(topo.cell_edge_signs == 1)


def test_build_edge_topology_two_tetra_has_negative_sign_due_to_local_orientation():
    mesh = make_two_tetra_mesh()
    topo = build_edge_topology(mesh)

    assert topo.cell_to_global_edges.shape == (2, 6)
    assert topo.cell_edge_signs.shape == (2, 6)
    assert np.any(topo.cell_edge_signs[1] == -1)


def test_boundary_edges_single_tetra():
    mesh = make_single_tetra_mesh()
    bedges = boundary_edges(mesh)

    assert len(bedges) == 6


def test_boundary_edges_two_tetra():
    mesh = make_two_tetra_mesh()
    bedges = boundary_edges(mesh)

    assert len(bedges) == 9


def test_interior_edges_two_tetra_empty_for_face_sharing_pair():
    mesh = make_two_tetra_mesh()
    iedges = interior_edges(mesh)

    assert iedges == ()