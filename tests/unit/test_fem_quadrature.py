from __future__ import annotations

import numpy as np

from magcore.femcore.quadrature import (
    get_tetra_quadrature,
    tetra_quadrature_1pt,
    tetra_quadrature_4pt,
    tetra_quadrature_5pt,
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


def test_tetra_quadrature_5pt_basic_properties():
    q = tetra_quadrature_5pt()

    assert q.n_points == 5
    assert q.points.shape == (5, 3)
    assert q.weights.shape == (5,)
    assert np.isclose(q.weight_sum, 1.0 / 6.0)

    # Для правила порядка 3 допустим отрицательный центральный вес.
    assert np.any(q.weights < 0.0)


def test_tetra_quadrature_points_are_inside_reference_tetra():
    for q in (tetra_quadrature_1pt(), tetra_quadrature_4pt(), tetra_quadrature_5pt()):
        for p in q.points:
            assert np.all(p >= 0.0)
            assert np.sum(p) <= 1.0 + 1e-14


def test_get_tetra_quadrature_order_1():
    q = get_tetra_quadrature(order=1)

    assert q.n_points == 1
    assert np.isclose(q.weight_sum, 1.0 / 6.0)


def test_get_tetra_quadrature_order_2():
    q = get_tetra_quadrature(order=2)

    assert q.n_points == 4
    assert np.isclose(q.weight_sum, 1.0 / 6.0)


def test_get_tetra_quadrature_order_3():
    q = get_tetra_quadrature(order=3)

    assert q.n_points == 5
    assert np.isclose(q.weight_sum, 1.0 / 6.0)


def test_get_tetra_quadrature_rejects_unsupported_order():
    try:
        get_tetra_quadrature(order=4)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for unsupported tetra quadrature order."