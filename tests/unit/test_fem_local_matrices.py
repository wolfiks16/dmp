from __future__ import annotations

import numpy as np

from magcore.femcore.local_matrices import (
    local_curlcurl_matrix,
    local_mass_matrix,
    local_rhs_vector,
)
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


def constant_rhs(x: np.ndarray) -> np.ndarray:
    return np.array([1.0, -2.0, 0.5], dtype=float)


def test_local_curlcurl_matrix_shape_and_finiteness():
    mesh = make_single_tetra_mesh()
    K = local_curlcurl_matrix(mesh, cell_idx=0, nu=2.0, quadrature_order=1)

    assert K.shape == (6, 6)
    assert np.isfinite(K).all()


def test_local_curlcurl_matrix_is_symmetric():
    mesh = make_single_tetra_mesh()
    K = local_curlcurl_matrix(mesh, cell_idx=0, nu=1.0, quadrature_order=1)

    assert np.allclose(K, K.T, atol=1.0e-12, rtol=1.0e-12)


def test_local_mass_matrix_shape_and_finiteness():
    mesh = make_single_tetra_mesh()
    M = local_mass_matrix(mesh, cell_idx=0, alpha=0.5, quadrature_order=2)

    assert M.shape == (6, 6)
    assert np.isfinite(M).all()


def test_local_mass_matrix_is_symmetric():
    mesh = make_single_tetra_mesh()
    M = local_mass_matrix(mesh, cell_idx=0, alpha=1.0, quadrature_order=2)

    assert np.allclose(M, M.T, atol=1.0e-12, rtol=1.0e-12)


def test_local_mass_matrix_is_positive_definite():
    mesh = make_single_tetra_mesh()
    M = local_mass_matrix(mesh, cell_idx=0, alpha=1.0, quadrature_order=2)

    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0.0)


def test_local_curlcurl_plus_mass_is_positive_definite():
    mesh = make_single_tetra_mesh()
    K = local_curlcurl_matrix(mesh, cell_idx=0, nu=1.0, quadrature_order=1)
    M = local_mass_matrix(mesh, cell_idx=0, alpha=0.1, quadrature_order=2)

    A = K + M
    eigvals = np.linalg.eigvalsh(A)
    assert np.all(eigvals > 0.0)


def test_local_rhs_vector_shape_and_finiteness():
    mesh = make_single_tetra_mesh()
    F = local_rhs_vector(mesh, cell_idx=0, f_fn=constant_rhs, quadrature_order=2)

    assert F.shape == (6,)
    assert np.isfinite(F).all()


def test_local_rhs_vector_is_linear_in_source():
    mesh = make_single_tetra_mesh()

    F1 = local_rhs_vector(mesh, cell_idx=0, f_fn=lambda x: constant_rhs(x), quadrature_order=2)
    F2 = local_rhs_vector(mesh, cell_idx=0, f_fn=lambda x: 2.0 * constant_rhs(x), quadrature_order=2)

    assert np.allclose(F2, 2.0 * F1)


def test_local_curlcurl_rejects_nonpositive_nu():
    mesh = make_single_tetra_mesh()

    try:
        local_curlcurl_matrix(mesh, cell_idx=0, nu=0.0, quadrature_order=1)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for nonpositive nu."


def test_local_mass_rejects_nonpositive_alpha():
    mesh = make_single_tetra_mesh()

    try:
        local_mass_matrix(mesh, cell_idx=0, alpha=0.0, quadrature_order=2)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for nonpositive alpha."


def test_local_rhs_vector_rejects_wrong_rhs_shape():
    mesh = make_single_tetra_mesh()

    try:
        local_rhs_vector(
            mesh,
            cell_idx=0,
            f_fn=lambda x: np.array([1.0, 2.0], dtype=float),
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong rhs vector shape."