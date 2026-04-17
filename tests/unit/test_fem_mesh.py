from __future__ import annotations

import numpy as np

from magcore.mesh.mesh import (
    TetraMesh,
    canonical_cell,
    canonical_edge,
    canonical_face,
)


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


def test_canonical_edge_face_cell_are_sorted() -> None:
    assert canonical_edge((3, 1)) == (1, 3)
    assert canonical_face((5, 1, 3)) == (1, 3, 5)
    assert canonical_cell((4, 2, 1, 3)) == (1, 2, 3, 4)


def test_tetra_mesh_basic_properties():
    mesh = make_single_tetra_mesh()

    assert mesh.n_vertices == 4
    assert mesh.n_cells == 1


def test_tetra_mesh_cell_vertex_indices_and_vertices():
    mesh = make_single_tetra_mesh()

    assert mesh.cell_vertex_indices(0) == (0, 1, 2, 3)
    verts = mesh.cell_vertices(0)

    assert verts.shape == (4, 3)
    assert np.allclose(verts[0], [0.0, 0.0, 0.0])


def test_tetra_mesh_cell_volume_is_correct():
    mesh = make_single_tetra_mesh()

    assert np.isclose(mesh.cell_volume(0), 1.0 / 6.0)


def test_tetra_mesh_cell_centroid_is_correct():
    mesh = make_single_tetra_mesh()

    c = mesh.cell_centroid(0)
    assert np.allclose(c, [0.25, 0.25, 0.25])


def test_tetra_mesh_cell_edges_single_tetra():
    mesh = make_single_tetra_mesh()

    edges = mesh.cell_edges(0)
    assert len(edges) == 6
    assert (0, 1) in edges
    assert (0, 2) in edges
    assert (0, 3) in edges
    assert (1, 2) in edges
    assert (1, 3) in edges
    assert (2, 3) in edges


def test_tetra_mesh_all_edges_and_edge_count():
    mesh = make_single_tetra_mesh()

    all_edges = mesh.all_edges()
    assert len(all_edges) == 6
    assert mesh.edge_count() == 6


def test_tetra_mesh_boundary_faces_single_tetra():
    mesh = make_single_tetra_mesh()
    bfaces = mesh.boundary_faces()

    assert len(bfaces) == 4
    assert (0, 1, 2) in bfaces
    assert (0, 1, 3) in bfaces
    assert (0, 2, 3) in bfaces
    assert (1, 2, 3) in bfaces


def test_tetra_mesh_boundary_faces_two_tetra():
    mesh = make_two_tetra_mesh()
    bfaces = mesh.boundary_faces()

    assert len(bfaces) == 6
    assert (0, 1, 2) not in bfaces


def test_tetra_mesh_boundary_edges_and_vertices_single_tetra():
    mesh = make_single_tetra_mesh()

    bedges = mesh.boundary_edges()
    bverts = mesh.boundary_vertices()

    assert len(bedges) == 6
    assert bverts == (0, 1, 2, 3)


def test_tetra_mesh_boundary_face_count():
    mesh = make_single_tetra_mesh()

    assert mesh.boundary_face_count() == 4


def test_tetra_mesh_rejects_nonpositive_orientation():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 2, 1, 3]], dtype=int)

    try:
        TetraMesh(vertices=vertices, cells=cells)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for non-positive oriented tetrahedron."


def test_tetra_mesh_rejects_repeated_vertex_indices():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 1, 2]], dtype=int)

    try:
        TetraMesh(vertices=vertices, cells=cells)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for repeated cell vertex indices."


def test_tetra_mesh_rejects_duplicate_tetrahedra():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=int,
    )

    try:
        TetraMesh(vertices=vertices, cells=cells)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for duplicate tetrahedra."


def test_tetra_mesh_rejects_nonmanifold_face():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # 0
            [1.0, 0.0, 0.0],   # 1
            [0.0, 1.0, 0.0],   # 2
            [0.0, 0.0, 1.0],   # 3
            [0.0, 0.0, -1.0],  # 4
            [1.0, 1.0, 1.0],   # 5
        ],
        dtype=float,
    )

    cells = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 4],
            [0, 1, 2, 5],
        ],
        dtype=int,
    )

    try:
        TetraMesh(vertices=vertices, cells=cells)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for a non-manifold face shared by > 2 cells."