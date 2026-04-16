from __future__ import annotations

import numpy as np

from magcore.femcore.quadrature import (
    get_tetra_quadrature,
    tetra_quadrature_1pt,
    tetra_quadrature_4pt,
)


def test_tetra_quadrature_1pt_basic_properties():
    q = tetra_quadrature_1pt()

    assert q.n_points == 1
    assert q.points.shape == (1, 3)
    assert q.weights.shape == (1,)
    assert np.isclose(q.weight_sum, 1.0 / 6.0)


def test_tetra_quadrature_4pt_basic_properties():
    q = tetra_quadrature_4pt()

    assert q.n_points == 4
    assert q.points.shape == (4, 3)
    assert q.weights.shape == (4,)
    assert np.isclose(q.weight_sum, 1.0 / 6.0)


def test_tetra_quadrature_points_are_inside_reference_tetra():
    q = tetra_quadrature_4pt()

    for p in q.points:
        assert np.all(p >= 0.0)
        assert np.sum(p) <= 1.0


def test_get_tetra_quadrature_order_1():
    q = get_tetra_quadrature(order=1)

    assert q.n_points == 1
    assert np.isclose(q.weight_sum, 1.0 / 6.0)


def test_get_tetra_quadrature_order_2():
    q = get_tetra_quadrature(order=2)

    assert q.n_points == 4
    assert np.isclose(q.weight_sum, 1.0 / 6.0)


def test_get_tetra_quadrature_rejects_unsupported_order():
    try:
        get_tetra_quadrature(order=3)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for unsupported tetra quadrature order."