from __future__ import annotations

import numpy as np

from magcore.femcore.mesh import TetraMesh
from magcore.femcore.scalar_spaces import LagrangeP1Space


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


def test_lagrange_p1_space_single_tetra_basic_properties():
    mesh = make_single_tetra_mesh()
    space = LagrangeP1Space(mesh=mesh)

    assert space.ndofs == 4
    assert space.cell_to_global_vertices.shape == (1, 4)


def test_lagrange_p1_space_two_tetra_basic_properties():
    mesh = make_two_tetra_mesh()
    space = LagrangeP1Space(mesh=mesh)

    assert space.ndofs == 5
    assert space.cell_to_global_vertices.shape == (2, 4)


def test_lagrange_p1_space_cell_dof_indices():
    mesh = make_single_tetra_mesh()
    space = LagrangeP1Space(mesh=mesh)

    dofs = space.cell_dof_indices(0)

    assert dofs.shape == (4,)
    assert tuple(dofs.tolist()) == (0, 1, 2, 3)


def test_lagrange_p1_space_boundary_vertex_indices_single_tetra():
    mesh = make_single_tetra_mesh()
    space = LagrangeP1Space(mesh=mesh)

    bverts = space.boundary_vertex_indices()

    assert bverts == (0, 1, 2, 3)


def test_lagrange_p1_space_boundary_vertex_indices_two_tetra():
    mesh = make_two_tetra_mesh()
    space = LagrangeP1Space(mesh=mesh)

    bverts = space.boundary_vertex_indices()

    assert bverts == (0, 1, 2, 3, 4)


def test_lagrange_p1_space_boundary_dofs_equal_boundary_vertices():
    mesh = make_two_tetra_mesh()
    space = LagrangeP1Space(mesh=mesh)

    assert space.boundary_dofs() == space.boundary_vertex_indices()