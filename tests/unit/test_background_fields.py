from __future__ import annotations

import numpy as np

from magcore.bem.background_fields import (
    evaluate_background_on_points,
    linear_background_gradient,
    linear_background_normal_flux,
    linear_background_potential,
)


def test_linear_background_potential_matches_minus_dot_product():
    H0 = np.array([1.0, -2.0, 0.5], dtype=float)
    phi = linear_background_potential(H0)

    x = np.array([3.0, 4.0, -1.0], dtype=float)
    expected = -np.dot(H0, x)

    assert np.isclose(phi(x), expected)


def test_linear_background_gradient_is_minus_H0():
    H0 = np.array([1.5, -0.25, 2.0], dtype=float)

    grad_phi = linear_background_gradient(H0)

    assert grad_phi.shape == (3,)
    assert np.allclose(grad_phi, -H0)


def test_linear_background_normal_flux_matches_gradient_dot_normal():
    H0 = np.array([0.0, 0.0, 2.0], dtype=float)
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    q = linear_background_normal_flux(H0, normals)

    expected = np.array([-2.0, 2.0, 0.0], dtype=float)
    assert q.shape == (3,)
    assert np.allclose(q, expected)


def test_evaluate_background_on_points_returns_expected_shape():
    H0 = np.array([1.0, 0.0, 0.0], dtype=float)
    phi = linear_background_potential(H0)

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 4.0],
        ],
        dtype=float,
    )

    vals = evaluate_background_on_points(points, phi)

    assert vals.shape == (3,)
    assert np.allclose(vals, np.array([0.0, -1.0, 1.0], dtype=float))


def test_linear_background_normal_flux_checks_shape():
    H0 = np.array([1.0, 0.0, 0.0], dtype=float)
    normals = np.array([1.0, 0.0, 0.0], dtype=float)

    try:
        linear_background_normal_flux(H0, normals)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for normals with wrong shape."


def test_linear_background_potential_checks_H0_shape():
    H0 = np.array([1.0, 2.0], dtype=float)

    try:
        linear_background_potential(H0)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for H0 with wrong shape."