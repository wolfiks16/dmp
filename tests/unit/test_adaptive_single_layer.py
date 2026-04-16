from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import (
    AdaptiveIntegrationConfig,
    terminal_pair_approximation,
    single_layer_triangle_pair_adaptive,
    single_layer_triangle_self_adaptive,
    single_layer_face_face_full,
)
from magcore.mesh.surface_mesh import SurfaceMesh


def make_separated_triangles():
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
    return tri_a, tri_b


def make_touching_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
            [2.0, 0.0, 0.0],  # 4
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],  # face 0
            [1, 3, 2],  # face 1 shared-edge with face 0
            [1, 4, 3],  # face 2 shared-vertex with face 0
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def make_separated_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_terminal_pair_approximation_is_positive():
    tri_a, tri_b = make_separated_triangles()
    val = terminal_pair_approximation(tri_a, tri_b)

    assert np.isfinite(val)
    assert val > 0.0


def test_single_layer_triangle_pair_adaptive_is_positive():
    tri_a, tri_b = make_separated_triangles()
    config = AdaptiveIntegrationConfig(quadrature_order=2, near_factor=1.0, max_depth=4)

    val = single_layer_triangle_pair_adaptive(tri_a, tri_b, config=config)
    assert np.isfinite(val)
    assert val > 0.0


def test_single_layer_triangle_pair_adaptive_is_symmetric():
    tri_a, tri_b = make_separated_triangles()
    config = AdaptiveIntegrationConfig(quadrature_order=2, near_factor=1.0, max_depth=4)

    vab = single_layer_triangle_pair_adaptive(tri_a, tri_b, config=config)
    vba = single_layer_triangle_pair_adaptive(tri_b, tri_a, config=config)

    assert np.isclose(vab, vba, rtol=1e-12, atol=1e-12)


def test_single_layer_triangle_self_adaptive_is_finite_and_positive():
    tri_a, _ = make_separated_triangles()
    config = AdaptiveIntegrationConfig(quadrature_order=2, self_max_depth=5, max_depth=4)

    val = single_layer_triangle_self_adaptive(tri_a, config=config)

    assert np.isfinite(val)
    assert val > 0.0


def test_single_layer_face_face_full_regular_pair_is_finite():
    mesh = make_separated_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, near_factor=1.0, max_depth=4)

    val = single_layer_face_face_full(mesh, 0, 1, config=config)

    assert np.isfinite(val)
    assert val > 0.0


def test_single_layer_face_face_full_self_is_finite():
    mesh = make_touching_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, self_max_depth=5, max_depth=4)

    val = single_layer_face_face_full(mesh, 0, 0, config=config)

    assert np.isfinite(val)
    assert val > 0.0


def test_single_layer_face_face_full_shared_edge_is_finite():
    mesh = make_touching_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, max_depth=4, self_max_depth=5)

    val = single_layer_face_face_full(mesh, 0, 1, config=config)

    assert np.isfinite(val)
    assert val > 0.0


def test_single_layer_face_face_full_shared_vertex_is_finite():
    mesh = make_touching_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, max_depth=4, self_max_depth=5)

    val = single_layer_face_face_full(mesh, 0, 2, config=config)

    assert np.isfinite(val)
    assert val > 0.0