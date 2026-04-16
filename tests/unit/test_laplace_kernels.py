from __future__ import annotations

import numpy as np

from magcore.bem.laplace_kernels import laplace_green_3d


def test_laplace_green_is_symmetric():
    x = np.array([0.0, 0.0, 0.0], dtype=float)
    y = np.array([1.0, 2.0, 3.0], dtype=float)

    assert np.isclose(laplace_green_3d(x, y), laplace_green_3d(y, x))


def test_laplace_green_is_positive():
    x = np.array([0.0, 0.0, 0.0], dtype=float)
    y = np.array([1.0, 0.0, 0.0], dtype=float)

    assert laplace_green_3d(x, y) > 0.0


def test_laplace_green_decays_with_distance():
    x = np.array([0.0, 0.0, 0.0], dtype=float)
    y1 = np.array([1.0, 0.0, 0.0], dtype=float)
    y2 = np.array([2.0, 0.0, 0.0], dtype=float)

    assert laplace_green_3d(x, y1) > laplace_green_3d(x, y2)


def test_laplace_green_raises_on_coincident_points():
    x = np.array([0.0, 0.0, 0.0], dtype=float)

    try:
        laplace_green_3d(x, x)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for coincident points."