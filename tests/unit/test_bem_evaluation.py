from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.bem.evaluation import evaluate_single_layer_potential_p0


def make_single_triangle_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_evaluate_single_layer_potential_shape():
    mesh = make_single_triangle_mesh()
    face_indices = (0,)
    density = np.array([1.0], dtype=float)
    target_points = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=float,
    )

    phi = evaluate_single_layer_potential_p0(
        mesh=mesh,
        face_indices=face_indices,
        density=density,
        target_points=target_points,
        quadrature_order=2,
    )

    assert phi.shape == (2,)


def test_evaluate_single_layer_potential_positive():
    mesh = make_single_triangle_mesh()
    phi = evaluate_single_layer_potential_p0(
        mesh=mesh,
        face_indices=(0,),
        density=np.array([1.0], dtype=float),
        target_points=np.array([[0.0, 0.0, 5.0]], dtype=float),
        quadrature_order=2,
    )

    assert phi[0] > 0.0


def test_evaluate_single_layer_potential_decays_with_distance():
    mesh = make_single_triangle_mesh()
    phi = evaluate_single_layer_potential_p0(
        mesh=mesh,
        face_indices=(0,),
        density=np.array([1.0], dtype=float),
        target_points=np.array(
            [
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=float,
        ),
        quadrature_order=2,
    )

    assert phi[0] > phi[1]


def test_evaluate_single_layer_potential_is_linear_in_density():
    mesh = make_single_triangle_mesh()
    pts = np.array([[0.0, 0.0, 5.0]], dtype=float)

    phi1 = evaluate_single_layer_potential_p0(
        mesh=mesh,
        face_indices=(0,),
        density=np.array([1.0], dtype=float),
        target_points=pts,
        quadrature_order=2,
    )
    phi2 = evaluate_single_layer_potential_p0(
        mesh=mesh,
        face_indices=(0,),
        density=np.array([2.0], dtype=float),
        target_points=pts,
        quadrature_order=2,
    )

    assert np.isclose(phi2[0], 2.0 * phi1[0])


def test_evaluate_single_layer_potential_checks_density_shape():
    mesh = make_single_triangle_mesh()
    pts = np.array([[0.0, 0.0, 5.0]], dtype=float)

    try:
        evaluate_single_layer_potential_p0(
            mesh=mesh,
            face_indices=(0,),
            density=np.array([1.0, 2.0], dtype=float),
            target_points=pts,
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong density shape."


def test_evaluate_single_layer_potential_checks_target_points_shape():
    mesh = make_single_triangle_mesh()

    try:
        evaluate_single_layer_potential_p0(
            mesh=mesh,
            face_indices=(0,),
            density=np.array([1.0], dtype=float),
            target_points=np.array([0.0, 0.0, 5.0], dtype=float),
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong target_points shape."