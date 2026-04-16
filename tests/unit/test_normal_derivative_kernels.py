from __future__ import annotations

import numpy as np

from magcore.bem.normal_derivative_kernels import (
    laplace_dgreen_dn_x,
    laplace_dgreen_dn_y,
)


def test_laplace_dgreen_dn_x_is_finite_for_distinct_points():
    x = np.array([1.0, 0.0, 0.0], dtype=float)
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    n_x = np.array([1.0, 0.0, 0.0], dtype=float)

    val = laplace_dgreen_dn_x(x, y, n_x)

    assert np.isfinite(val)


def test_laplace_dgreen_dn_y_is_finite_for_distinct_points():
    x = np.array([1.0, 0.0, 0.0], dtype=float)
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    n_y = np.array([1.0, 0.0, 0.0], dtype=float)

    val = laplace_dgreen_dn_y(x, y, n_y)

    assert np.isfinite(val)


def test_laplace_dgreen_dn_x_changes_sign_with_normal_flip():
    x = np.array([2.0, 0.0, 0.0], dtype=float)
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    n_x = np.array([1.0, 0.0, 0.0], dtype=float)

    v1 = laplace_dgreen_dn_x(x, y, n_x)
    v2 = laplace_dgreen_dn_x(x, y, -n_x)

    assert np.isclose(v2, -v1)


def test_laplace_dgreen_dn_y_changes_sign_with_normal_flip():
    x = np.array([2.0, 0.0, 0.0], dtype=float)
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    n_y = np.array([1.0, 0.0, 0.0], dtype=float)

    v1 = laplace_dgreen_dn_y(x, y, n_y)
    v2 = laplace_dgreen_dn_y(x, y, -n_y)

    assert np.isclose(v2, -v1)


def test_laplace_dgreen_dn_x_decays_with_distance():
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    n_x = np.array([1.0, 0.0, 0.0], dtype=float)

    x1 = np.array([1.0, 0.0, 0.0], dtype=float)
    x2 = np.array([2.0, 0.0, 0.0], dtype=float)

    v1 = abs(laplace_dgreen_dn_x(x1, y, n_x))
    v2 = abs(laplace_dgreen_dn_x(x2, y, n_x))

    assert v1 > v2


def test_laplace_dgreen_dn_y_decays_with_distance():
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    n_y = np.array([1.0, 0.0, 0.0], dtype=float)

    x1 = np.array([1.0, 0.0, 0.0], dtype=float)
    x2 = np.array([2.0, 0.0, 0.0], dtype=float)

    v1 = abs(laplace_dgreen_dn_y(x1, y, n_y))
    v2 = abs(laplace_dgreen_dn_y(x2, y, n_y))

    assert v1 > v2


def test_laplace_dgreen_dn_x_matches_negative_of_dn_y_for_same_normal():
    x = np.array([2.0, 0.0, 0.0], dtype=float)
    y = np.array([0.0, 0.0, 0.0], dtype=float)
    n = np.array([1.0, 0.0, 0.0], dtype=float)

    vx = laplace_dgreen_dn_x(x, y, n)
    vy = laplace_dgreen_dn_y(x, y, n)

    assert np.isclose(vx, -vy)


def test_laplace_dgreen_dn_x_raises_for_coincident_points():
    x = np.array([0.0, 0.0, 0.0], dtype=float)
    n_x = np.array([1.0, 0.0, 0.0], dtype=float)

    try:
        laplace_dgreen_dn_x(x, x, n_x)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for coincident points."


def test_laplace_dgreen_dn_y_raises_for_coincident_points():
    x = np.array([0.0, 0.0, 0.0], dtype=float)
    n_y = np.array([1.0, 0.0, 0.0], dtype=float)

    try:
        laplace_dgreen_dn_y(x, x, n_y)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for coincident points."