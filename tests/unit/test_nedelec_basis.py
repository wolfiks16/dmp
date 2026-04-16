from __future__ import annotations

import numpy as np

from magcore.femcore.basis_nedelec import (
    edge_line_integral_of_basis_on_straight_edge,
    local_edge_tangent_vector,
    physical_nedelec_basis,
    physical_nedelec_curl,
    reference_nedelec_basis,
    reference_nedelec_curl,
)
from magcore.femcore.reference_tetra import AffineTetraMap, REFERENCE_TETRA_VERTICES


def make_identity_map() -> AffineTetraMap:
    return AffineTetraMap(physical_vertices=REFERENCE_TETRA_VERTICES.copy())


def make_scaled_map() -> AffineTetraMap:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    return AffineTetraMap(physical_vertices=verts)


def test_reference_nedelec_basis_returns_vector():
    p = np.array([0.2, 0.3, 0.1], dtype=float)

    for e in range(6):
        w = reference_nedelec_basis(e, p)
        assert w.shape == (3,)
        assert np.isfinite(w).all()


def test_reference_nedelec_curl_returns_vector():
    for e in range(6):
        c = reference_nedelec_curl(e)
        assert c.shape == (3,)
        assert np.isfinite(c).all()


def test_reference_nedelec_curl_is_constant():
    for e in range(6):
        c1 = reference_nedelec_curl(e)
        c2 = reference_nedelec_curl(e)
        assert np.allclose(c1, c2)


def test_physical_nedelec_basis_identity_map_matches_reference():
    amap = make_identity_map()
    x = np.array([0.2, 0.3, 0.1], dtype=float)

    for e in range(6):
        w_phys = physical_nedelec_basis(amap, e, x)
        w_ref = reference_nedelec_basis(e, x)
        assert np.allclose(w_phys, w_ref)


def test_physical_nedelec_curl_identity_map_matches_reference():
    amap = make_identity_map()

    for e in range(6):
        c_phys = physical_nedelec_curl(amap, e)
        c_ref = reference_nedelec_curl(e)
        assert np.allclose(c_phys, c_ref)


def test_physical_nedelec_basis_scaled_map_is_finite():
    amap = make_scaled_map()
    x = np.array([0.2, 0.3, 0.4], dtype=float)

    for e in range(6):
        w = physical_nedelec_basis(amap, e, x)
        assert w.shape == (3,)
        assert np.isfinite(w).all()


def test_physical_nedelec_curl_scaled_map_is_finite():
    amap = make_scaled_map()

    for e in range(6):
        c = physical_nedelec_curl(amap, e)
        assert c.shape == (3,)
        assert np.isfinite(c).all()


def test_local_edge_tangent_vector_is_correct():
    amap = make_identity_map()

    tau = local_edge_tangent_vector(amap, 0)  # edge (0,1)
    assert np.allclose(tau, [1.0, 0.0, 0.0])

    tau = local_edge_tangent_vector(amap, 5)  # edge (2,3)
    assert np.allclose(tau, [0.0, -1.0, 1.0])


def test_edge_line_integrals_give_kronecker_structure_identity_map():
    amap = make_identity_map()

    for basis_edge in range(6):
        for test_edge in range(6):
            val = edge_line_integral_of_basis_on_straight_edge(
                affine_map=amap,
                basis_edge_idx=basis_edge,
                test_edge_idx=test_edge,
                n_points=16,
            )

            if basis_edge == test_edge:
                assert np.isclose(val, 1.0, atol=1.0e-10)
            else:
                assert np.isclose(val, 0.0, atol=1.0e-10)


def test_edge_line_integrals_give_kronecker_structure_scaled_map():
    amap = make_scaled_map()

    for basis_edge in range(6):
        for test_edge in range(6):
            val = edge_line_integral_of_basis_on_straight_edge(
                affine_map=amap,
                basis_edge_idx=basis_edge,
                test_edge_idx=test_edge,
                n_points=16,
            )

            if basis_edge == test_edge:
                assert np.isclose(val, 1.0, atol=1.0e-9)
            else:
                assert np.isclose(val, 0.0, atol=1.0e-9)


def test_reference_nedelec_basis_checks_edge_index():
    p = np.array([0.2, 0.2, 0.2], dtype=float)

    try:
        reference_nedelec_basis(6, p)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for invalid edge index."


def test_reference_nedelec_curl_checks_edge_index():
    try:
        reference_nedelec_curl(6)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for invalid edge index."


def test_physical_nedelec_basis_checks_point_shape():
    amap = make_identity_map()

    try:
        physical_nedelec_basis(amap, 0, np.array([0.1, 0.2], dtype=float))
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong point shape."