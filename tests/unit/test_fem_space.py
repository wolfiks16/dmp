from __future__ import annotations

import numpy as np

from magcore.femcore.mesh import TetraMesh
from magcore.femcore.spaces import NedelecP1Space


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


def test_nedelec_space_single_tetra_basic_properties():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    assert space.ndofs == 6
    assert space.cell_to_global_edges.shape == (1, 6)
    assert space.cell_edge_signs.shape == (1, 6)


def test_nedelec_space_two_tetra_basic_properties():
    mesh = make_two_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    assert space.ndofs == 9
    assert space.cell_to_global_edges.shape == (2, 6)
    assert space.cell_edge_signs.shape == (2, 6)


def test_nedelec_space_cell_dof_indices_and_signs():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    dofs = space.cell_dof_indices(0)
    sgn = space.cell_dof_signs(0)

    assert dofs.shape == (6,)
    assert sgn.shape == (6,)
    assert np.all(sgn == 1)


def test_nedelec_space_edge_to_dof_map_size():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    mp = space.edge_to_dof_map()

    assert len(mp) == 6
    assert (0, 1) in mp
    assert (2, 3) in mp


def test_nedelec_space_boundary_dofs_single_tetra():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    bdofs = space.boundary_dofs()

    assert bdofs == (0, 1, 2, 3, 4, 5)


def test_nedelec_space_boundary_dofs_two_tetra():
    mesh = make_two_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    bdofs = space.boundary_dofs()

    assert len(bdofs) == 9
    assert tuple(sorted(bdofs)) == tuple(range(9))