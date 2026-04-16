from __future__ import annotations

import numpy as np

from magcore.femcore.mesh import TetraMesh


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
    assert (0, 1, 2) not in bfaces  # shared internal face


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
    cells = np.array([[0, 2, 1, 3]], dtype=int)  # flipped orientation

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