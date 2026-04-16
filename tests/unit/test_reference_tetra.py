from __future__ import annotations

import numpy as np

from magcore.femcore.reference_tetra import (
    AffineTetraMap,
    REFERENCE_TETRA_VERTICES,
    reference_barycentric,
    reference_barycentric_gradients,
)


def test_reference_barycentric_sums_to_one():
    p = np.array([0.2, 0.3, 0.1], dtype=float)
    lam = reference_barycentric(p)

    assert lam.shape == (4,)
    assert np.isclose(np.sum(lam), 1.0)


def test_reference_barycentric_at_reference_vertices():
    expected = np.eye(4, dtype=float)

    for i in range(4):
        lam = reference_barycentric(REFERENCE_TETRA_VERTICES[i])
        assert np.allclose(lam, expected[i])


def test_reference_barycentric_gradients_are_correct():
    grads = reference_barycentric_gradients()

    expected = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    assert grads.shape == (4, 3)
    assert np.allclose(grads, expected)


def test_affine_tetra_map_identity_case():
    amap = AffineTetraMap(physical_vertices=REFERENCE_TETRA_VERTICES.copy())

    assert np.isclose(amap.jacobian_determinant(), 1.0)
    assert np.isclose(amap.volume(), 1.0 / 6.0)

    p_ref = np.array([0.2, 0.3, 0.1], dtype=float)
    p_phys = amap.map_to_physical(p_ref)

    assert np.allclose(p_phys, p_ref)
    assert np.allclose(amap.map_to_reference(p_phys), p_ref)


def test_affine_tetra_map_scaled_case():
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    amap = AffineTetraMap(physical_vertices=verts)

    assert np.isclose(amap.jacobian_determinant(), 24.0)
    assert np.isclose(amap.volume(), 4.0)


def test_affine_tetra_map_barycentric_at_physical_point():
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    amap = AffineTetraMap(physical_vertices=verts)

    p_ref = np.array([0.2, 0.3, 0.1], dtype=float)
    p_phys = amap.map_to_physical(p_ref)

    lam = amap.barycentric_at_physical_point(p_phys)
    assert np.allclose(lam, reference_barycentric(p_ref))


def test_affine_tetra_map_physical_barycentric_gradients_scaled_case():
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    amap = AffineTetraMap(physical_vertices=verts)

    grads = amap.physical_barycentric_gradients()

    expected = np.array(
        [
            [-0.5, -1.0 / 3.0, -0.25],
            [0.5, 0.0, 0.0],
            [0.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.25],
        ],
        dtype=float,
    )

    assert grads.shape == (4, 3)
    assert np.allclose(grads, expected)


def test_affine_tetra_map_rejects_nonpositive_orientation():
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=float,
    )

    try:
        AffineTetraMap(physical_vertices=verts)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for non-positive oriented physical tetrahedron."