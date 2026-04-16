from __future__ import annotations

import numpy as np

from magcore.femcore.mesh import TetraMesh
from magcore.femcore.mixed_local_matrices import (
    local_curlcurl_block,
    local_grad_p_coupling_matrix,
    local_scalar_mass_matrix,
    local_vector_source_rhs,
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


def constant_current_density(x: np.ndarray) -> np.ndarray:
    return np.array([1.0, -2.0, 0.5], dtype=float)


def test_local_vector_source_rhs_shape_and_finiteness():
    mesh = make_single_tetra_mesh()

    F = local_vector_source_rhs(
        mesh=mesh,
        cell_idx=0,
        J_fn=constant_current_density,
        quadrature_order=2,
    )

    assert F.shape == (6,)
    assert np.isfinite(F).all()


def test_local_vector_source_rhs_is_linear_in_source():
    mesh = make_single_tetra_mesh()

    F1 = local_vector_source_rhs(
        mesh=mesh,
        cell_idx=0,
        J_fn=constant_current_density,
        quadrature_order=2,
    )
    F2 = local_vector_source_rhs(
        mesh=mesh,
        cell_idx=0,
        J_fn=lambda x: 2.0 * constant_current_density(x),
        quadrature_order=2,
    )

    assert np.allclose(F2, 2.0 * F1)


def test_local_vector_source_rhs_rejects_wrong_shape():
    mesh = make_single_tetra_mesh()

    try:
        local_vector_source_rhs(
            mesh=mesh,
            cell_idx=0,
            J_fn=lambda x: np.array([1.0, 2.0], dtype=float),
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong source shape."


def test_local_grad_p_coupling_matrix_shape_and_finiteness():
    mesh = make_single_tetra_mesh()

    G = local_grad_p_coupling_matrix(
        mesh=mesh,
        cell_idx=0,
        quadrature_order=2,
    )

    assert G.shape == (6, 4)
    assert np.isfinite(G).all()


def test_local_grad_p_coupling_matrix_is_nontrivial():
    mesh = make_single_tetra_mesh()

    G = local_grad_p_coupling_matrix(
        mesh=mesh,
        cell_idx=0,
        quadrature_order=2,
    )

    assert np.linalg.norm(G) > 0.0


def test_local_scalar_mass_matrix_shape_and_finiteness():
    mesh = make_single_tetra_mesh()

    M = local_scalar_mass_matrix(
        mesh=mesh,
        cell_idx=0,
        beta=1.0,
        quadrature_order=2,
    )

    assert M.shape == (4, 4)
    assert np.isfinite(M).all()


def test_local_scalar_mass_matrix_is_symmetric():
    mesh = make_single_tetra_mesh()

    M = local_scalar_mass_matrix(
        mesh=mesh,
        cell_idx=0,
        beta=1.0,
        quadrature_order=2,
    )

    assert np.allclose(M, M.T, atol=1.0e-12, rtol=1.0e-12)


def test_local_scalar_mass_matrix_is_positive_definite():
    mesh = make_single_tetra_mesh()

    M = local_scalar_mass_matrix(
        mesh=mesh,
        cell_idx=0,
        beta=1.0,
        quadrature_order=2,
    )

    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0.0)


def test_local_scalar_mass_matrix_rejects_nonpositive_beta():
    mesh = make_single_tetra_mesh()

    try:
        local_scalar_mass_matrix(
            mesh=mesh,
            cell_idx=0,
            beta=0.0,
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for nonpositive beta."


def test_local_curlcurl_block_matches_shape_and_finiteness():
    mesh = make_single_tetra_mesh()

    K = local_curlcurl_block(
        mesh=mesh,
        cell_idx=0,
        nu=1.0,
        quadrature_order=1,
    )

    assert K.shape == (6, 6)
    assert np.isfinite(K).all()