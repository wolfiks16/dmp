from __future__ import annotations

import numpy as np

from magcore.femcore.boundary_conditions import (
    apply_zero_dirichlet_bc,
    find_boundary_dofs,
)
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


def test_find_boundary_dofs_single_tetra_all_dofs_are_boundary():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    dofs = find_boundary_dofs(space)

    assert dofs == tuple(range(space.ndofs))


def test_find_boundary_dofs_two_tetra_all_dofs_are_boundary_for_face_pair():
    mesh = make_two_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    dofs = find_boundary_dofs(space)

    assert dofs == tuple(range(space.ndofs))


def test_apply_zero_dirichlet_bc_sets_identity_rows_and_columns():
    A = np.array(
        [
            [4.0, 1.0, 2.0, 3.0],
            [1.0, 5.0, 6.0, 7.0],
            [2.0, 6.0, 8.0, 9.0],
            [3.0, 7.0, 9.0, 10.0],
        ],
        dtype=float,
    )
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    Abc, bbc = apply_zero_dirichlet_bc(A, b, (1, 3))

    expected_row_1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    expected_row_3 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    assert np.allclose(Abc[1, :], expected_row_1)
    assert np.allclose(Abc[:, 1], expected_row_1)
    assert np.allclose(Abc[3, :], expected_row_3)
    assert np.allclose(Abc[:, 3], expected_row_3)

    assert np.isclose(bbc[1], 0.0)
    assert np.isclose(bbc[3], 0.0)


def test_apply_zero_dirichlet_bc_leaves_unconstrained_block_entries_intact():
    A = np.array(
        [
            [4.0, 1.0, 2.0, 3.0],
            [1.0, 5.0, 6.0, 7.0],
            [2.0, 6.0, 8.0, 9.0],
            [3.0, 7.0, 9.0, 10.0],
        ],
        dtype=float,
    )
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    Abc, bbc = apply_zero_dirichlet_bc(A, b, (1, 3))

    assert np.isclose(Abc[0, 0], 4.0)
    assert np.isclose(Abc[0, 2], 2.0)
    assert np.isclose(Abc[2, 0], 2.0)
    assert np.isclose(Abc[2, 2], 8.0)

    assert np.isclose(bbc[0], 1.0)
    assert np.isclose(bbc[2], 3.0)


def test_apply_zero_dirichlet_bc_accepts_empty_dof_set():
    A = np.array(
        [
            [2.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=float,
    )
    b = np.array([4.0, 5.0], dtype=float)

    Abc, bbc = apply_zero_dirichlet_bc(A, b, ())

    assert np.allclose(Abc, A)
    assert np.allclose(bbc, b)


def test_apply_zero_dirichlet_bc_rejects_nonsquare_matrix():
    A = np.ones((2, 3), dtype=float)
    b = np.ones(2, dtype=float)

    try:
        apply_zero_dirichlet_bc(A, b, (0,))
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for nonsquare matrix."


def test_apply_zero_dirichlet_bc_rejects_rhs_shape_mismatch():
    A = np.eye(3, dtype=float)
    b = np.ones(2, dtype=float)

    try:
        apply_zero_dirichlet_bc(A, b, (0,))
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for rhs shape mismatch."


def test_apply_zero_dirichlet_bc_rejects_out_of_range_dofs():
    A = np.eye(3, dtype=float)
    b = np.ones(3, dtype=float)

    try:
        apply_zero_dirichlet_bc(A, b, (3,))
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for out-of-range dof index."