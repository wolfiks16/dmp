from __future__ import annotations

import numpy as np

from magcore.bem.element_integrals import triangle_area
from magcore.bem.triangle_subdivision import (
    midpoint,
    subdivide_triangle_4,
    triangle_pair_distance_proxy,
    triangle_pair_is_regular,
    subdivision_area_conservation_error,
)


def make_triangle() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )


def test_midpoint_is_correct():
    a = np.array([0.0, 0.0, 0.0], dtype=float)
    b = np.array([2.0, 2.0, 2.0], dtype=float)

    m = midpoint(a, b)
    assert np.allclose(m, [1.0, 1.0, 1.0])


def test_subdivide_triangle_4_returns_four_children():
    tri = make_triangle()
    children = subdivide_triangle_4(tri)

    assert len(children) == 4
    assert all(child.shape == (3, 3) for child in children)


def test_subdivide_triangle_4_preserves_total_area():
    tri = make_triangle()
    children = subdivide_triangle_4(tri)

    parent_area = triangle_area(tri)
    child_area = sum(triangle_area(t) for t in children)

    assert np.isclose(parent_area, child_area)


def test_subdivide_triangle_4_children_are_non_degenerate():
    tri = make_triangle()
    children = subdivide_triangle_4(tri)

    for child in children:
        assert triangle_area(child) > 0.0


def test_subdivision_area_conservation_error_is_small():
    tri = make_triangle()
    err = subdivision_area_conservation_error(tri)

    assert np.isclose(err, 0.0)


def test_triangle_pair_distance_proxy_positive_for_separated_triangles():
    tri_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    tri_b = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )

    dist = triangle_pair_distance_proxy(tri_a, tri_b)
    assert dist > 0.0


def test_triangle_pair_is_regular_true_for_well_separated_triangles():
    tri_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    tri_b = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )

    assert triangle_pair_is_regular(tri_a, tri_b, near_factor=1.0) is True


def test_triangle_pair_is_regular_false_for_near_triangles():
    tri_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    tri_b = np.array(
        [
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
        ],
        dtype=float,
    )

    assert triangle_pair_is_regular(tri_a, tri_b, near_factor=2.0) is False