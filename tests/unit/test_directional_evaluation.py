from __future__ import annotations

import numpy as np

from magcore.bem.directional_evaluation import (
    estimate_zone_H_parallel_from_faces,
    evaluate_single_layer_directional_derivative_p0,
    normalize_directions,
    single_layer_directional_derivative_regular,
)
from magcore.mesh.surface_mesh import SurfaceMesh


def make_single_triangle_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return SurfaceMesh(vertices=vertices, faces=faces)


def make_source_triangle() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def test_normalize_directions_returns_unit_vectors():
    dirs = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, -3.0],
        ],
        dtype=float,
    )

    out = normalize_directions(dirs)

    assert out.shape == (2, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_normalize_directions_rejects_zero_vector():
    dirs = np.array([[0.0, 0.0, 0.0]], dtype=float)

    try:
        normalize_directions(dirs)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for zero direction vector."


def test_single_layer_directional_derivative_regular_is_finite():
    tri = make_source_triangle()
    x = np.array([0.0, 0.0, 5.0], dtype=float)
    d = np.array([0.0, 0.0, 1.0], dtype=float)

    val = single_layer_directional_derivative_regular(
        target_point=x,
        target_direction=d,
        source_tri=tri,
        quadrature_order=2,
    )

    assert np.isfinite(val)


def test_single_layer_directional_derivative_changes_sign_with_direction_flip():
    tri = make_source_triangle()
    x = np.array([0.0, 0.0, 5.0], dtype=float)
    d = np.array([0.0, 0.0, 1.0], dtype=float)

    v1 = single_layer_directional_derivative_regular(
        target_point=x,
        target_direction=d,
        source_tri=tri,
        quadrature_order=2,
    )
    v2 = single_layer_directional_derivative_regular(
        target_point=x,
        target_direction=-d,
        source_tri=tri,
        quadrature_order=2,
    )

    assert np.isclose(v2, -v1)


def test_evaluate_single_layer_directional_derivative_p0_shape():
    mesh = make_single_triangle_mesh()
    face_indices = (0,)
    density = np.array([1.0], dtype=float)
    pts = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=float,
    )
    dirs = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    vals = evaluate_single_layer_directional_derivative_p0(
        mesh=mesh,
        face_indices=face_indices,
        density=density,
        target_points=pts,
        target_directions=dirs,
        quadrature_order=2,
    )

    assert vals.shape == (2,)
    assert np.isfinite(vals).all()


def test_evaluate_single_layer_directional_derivative_decays_with_distance():
    mesh = make_single_triangle_mesh()
    face_indices = (0,)
    density = np.array([1.0], dtype=float)
    pts = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=float,
    )
    dirs = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    vals = evaluate_single_layer_directional_derivative_p0(
        mesh=mesh,
        face_indices=face_indices,
        density=density,
        target_points=pts,
        target_directions=dirs,
        quadrature_order=2,
    )

    assert abs(vals[0]) > abs(vals[1])


def test_evaluate_single_layer_directional_derivative_is_linear_in_density():
    mesh = make_single_triangle_mesh()
    face_indices = (0,)
    pts = np.array([[0.0, 0.0, 5.0]], dtype=float)
    dirs = np.array([[0.0, 0.0, 1.0]], dtype=float)

    v1 = evaluate_single_layer_directional_derivative_p0(
        mesh=mesh,
        face_indices=face_indices,
        density=np.array([1.0], dtype=float),
        target_points=pts,
        target_directions=dirs,
        quadrature_order=2,
    )
    v2 = evaluate_single_layer_directional_derivative_p0(
        mesh=mesh,
        face_indices=face_indices,
        density=np.array([2.0], dtype=float),
        target_points=pts,
        target_directions=dirs,
        quadrature_order=2,
    )

    assert np.isclose(v2[0], 2.0 * v1[0])


def test_evaluate_single_layer_directional_derivative_checks_density_shape():
    mesh = make_single_triangle_mesh()
    face_indices = (0,)
    pts = np.array([[0.0, 0.0, 5.0]], dtype=float)
    dirs = np.array([[0.0, 0.0, 1.0]], dtype=float)

    try:
        evaluate_single_layer_directional_derivative_p0(
            mesh=mesh,
            face_indices=face_indices,
            density=np.array([1.0, 2.0], dtype=float),
            target_points=pts,
            target_directions=dirs,
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong density shape."


def test_evaluate_single_layer_directional_derivative_checks_target_points_shape():
    mesh = make_single_triangle_mesh()
    face_indices = (0,)

    try:
        evaluate_single_layer_directional_derivative_p0(
            mesh=mesh,
            face_indices=face_indices,
            density=np.array([1.0], dtype=float),
            target_points=np.array([0.0, 0.0, 5.0], dtype=float),
            target_directions=np.array([[0.0, 0.0, 1.0]], dtype=float),
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong target_points shape."


def test_evaluate_single_layer_directional_derivative_checks_target_directions_shape():
    mesh = make_single_triangle_mesh()
    face_indices = (0,)
    pts = np.array([[0.0, 0.0, 5.0]], dtype=float)

    try:
        evaluate_single_layer_directional_derivative_p0(
            mesh=mesh,
            face_indices=face_indices,
            density=np.array([1.0], dtype=float),
            target_points=pts,
            target_directions=np.array([0.0, 0.0, 1.0], dtype=float),
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong target_directions shape."


def test_estimate_zone_H_parallel_from_faces_returns_negative_mean():
    vals = np.array([1.0, 2.0, 3.0], dtype=float)

    H_par = estimate_zone_H_parallel_from_faces(vals)

    assert np.isclose(H_par, -2.0)


def test_estimate_zone_H_parallel_from_faces_checks_finiteness():
    vals = np.array([1.0, np.nan], dtype=float)

    try:
        estimate_zone_H_parallel_from_faces(vals)
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for non-finite directional derivative values."