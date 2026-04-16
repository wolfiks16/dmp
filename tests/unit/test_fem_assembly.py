from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import (
    assemble_curlcurl_matrix,
    assemble_mass_matrix,
    assemble_rhs_vector,
    assemble_system,
)
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


def constant_rhs(x: np.ndarray) -> np.ndarray:
    return np.array([1.0, -2.0, 0.5], dtype=float)


def test_assemble_curlcurl_matrix_single_tetra():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    K = assemble_curlcurl_matrix(mesh, space, nu=1.0, quadrature_order=1)

    assert K.shape == (space.ndofs, space.ndofs)
    assert np.isfinite(K).all()
    assert np.allclose(K, K.T, atol=1.0e-12, rtol=1.0e-12)


def test_assemble_mass_matrix_single_tetra():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    M = assemble_mass_matrix(mesh, space, alpha=0.1, quadrature_order=2)

    assert M.shape == (space.ndofs, space.ndofs)
    assert np.isfinite(M).all()
    assert np.allclose(M, M.T, atol=1.0e-12, rtol=1.0e-12)


def test_assemble_rhs_vector_single_tetra():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    b = assemble_rhs_vector(mesh, space, f_fn=constant_rhs, quadrature_order=2)

    assert b.shape == (space.ndofs,)
    assert np.isfinite(b).all()


def test_assemble_system_single_tetra():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    A, b = assemble_system(
        mesh=mesh,
        space=space,
        nu=1.0,
        alpha=0.1,
        f_fn=constant_rhs,
        curl_quadrature_order=1,
        mass_quadrature_order=2,
        rhs_quadrature_order=2,
    )

    assert A.shape == (space.ndofs, space.ndofs)
    assert b.shape == (space.ndofs,)
    assert np.isfinite(A).all()
    assert np.isfinite(b).all()
    assert np.allclose(A, A.T, atol=1.0e-12, rtol=1.0e-12)


def test_assemble_system_two_tetra():
    mesh = make_two_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    A, b = assemble_system(
        mesh=mesh,
        space=space,
        nu=2.0,
        alpha=0.2,
        f_fn=constant_rhs,
        curl_quadrature_order=1,
        mass_quadrature_order=2,
        rhs_quadrature_order=2,
    )

    assert A.shape == (space.ndofs, space.ndofs)
    assert b.shape == (space.ndofs,)
    assert np.isfinite(A).all()
    assert np.isfinite(b).all()


def test_find_boundary_dofs_single_tetra():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    dofs = find_boundary_dofs(space)

    assert dofs == tuple(range(space.ndofs))


def test_apply_zero_dirichlet_bc_single_tetra():
    mesh = make_single_tetra_mesh()
    space = NedelecP1Space.from_mesh(mesh)

    A, b = assemble_system(
        mesh=mesh,
        space=space,
        nu=1.0,
        alpha=0.1,
        f_fn=constant_rhs,
    )
    dofs = find_boundary_dofs(space)
    Abc, bbc = apply_zero_dirichlet_bc(A, b, dofs)

    assert Abc.shape == A.shape
    assert bbc.shape == b.shape

    for d in dofs:
        row = Abc[d, :]
        col = Abc[:, d]
        expected = np.zeros(space.ndofs, dtype=float)
        expected[d] = 1.0

        assert np.allclose(row, expected)
        assert np.allclose(col, expected)
        assert np.isclose(bbc[d], 0.0)


def test_apply_zero_dirichlet_bc_rejects_bad_shapes():
    A = np.eye(4, dtype=float)
    b = np.ones(3, dtype=float)

    try:
        apply_zero_dirichlet_bc(A, b, (0, 1))
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for inconsistent A and b shapes."