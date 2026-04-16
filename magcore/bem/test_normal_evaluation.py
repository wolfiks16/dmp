from __future__ import annotations

import numpy as np

from magcore.bem.normal_evaluation import (
    assemble_single_layer_normal_trace_matrix_offset,
    evaluate_single_layer_normal_derivative_p0,
    face_centroids,
    face_characteristic_lengths,
    face_unit_normals,
    offset_face_centroids,
)
from magcore.mesh.surface_mesh import SurfaceMesh


def make_two_face_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 5.0],  # 3
            [1.0, 0.0, 5.0],  # 4
            [0.0, 1.0, 5.0],  # 5
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


def test_face_unit_normals_shape():
    mesh = make_two_face_mesh()
    normals = face_unit_normals(mesh, (0, 1))

    assert normals.shape == (2, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_face_characteristic_lengths_positive():
    mesh = make_two_face_mesh()
    h = face_characteristic_lengths(mesh, (0, 1))

    assert h.shape == (2,)
    assert np.all(h > 0.0)


def test_face_centroids_shape():
    mesh = make_two_face_mesh()
    ctrs = face_centroids(mesh, (0, 1))

    assert ctrs.shape == (2, 3)


def test_offset_face_centroids_shapes():
    mesh = make_two_face_mesh()
    ctrs, inner_pts, outer_pts = offset_face_centroids(mesh, (0, 1), offset_factor=0.05)

    assert ctrs.shape == (2, 3)
    assert inner_pts.shape == (2, 3)
    assert outer_pts.shape == (2, 3)


def test_offset_face_centroids_are_displaced():
    mesh = make_two_face_mesh()
    ctrs, inner_pts, outer_pts = offset_face_centroids(mesh, (0, 1), offset_factor=0.05)

    assert not np.allclose(ctrs, inner_pts)
    assert not np.allclose(ctrs, outer_pts)


def test_evaluate_single_layer_normal_derivative_p0_shape():
    mesh = make_two_face_mesh()
    face_indices = (0, 1)
    density = np.array([1.0, -0.5], dtype=float)

    _, _, outer_pts = offset_face_centroids(mesh, face_indices, offset_factor=0.05)
    normals = face_unit_normals(mesh, face_indices)

    vals = evaluate_single_layer_normal_derivative_p0(
        mesh=mesh,
        face_indices=face_indices,
        density=density,
        target_points=outer_pts,
        target_normals=normals,
        quadrature_order=2,
    )

    assert vals.shape == (2,)
    assert np.isfinite(vals).all()


def test_assemble_single_layer_normal_trace_matrix_offset_shape():
    mesh = make_two_face_mesh()
    face_indices = (0, 1)

    _, _, outer_pts = offset_face_centroids(mesh, face_indices, offset_factor=0.05)
    normals = face_unit_normals(mesh, face_indices)

    A = assemble_single_layer_normal_trace_matrix_offset(
        mesh=mesh,
        face_indices=face_indices,
        target_points=outer_pts,
        target_normals=normals,
        quadrature_order=2,
    )

    assert A.shape == (2, 2)
    assert np.isfinite(A).all()


def test_evaluate_single_layer_normal_derivative_checks_density_shape():
    mesh = make_two_face_mesh()
    face_indices = (0, 1)
    _, _, outer_pts = offset_face_centroids(mesh, face_indices, offset_factor=0.05)
    normals = face_unit_normals(mesh, face_indices)

    try:
        evaluate_single_layer_normal_derivative_p0(
            mesh=mesh,
            face_indices=face_indices,
            density=np.array([1.0], dtype=float),
            target_points=outer_pts,
            target_normals=normals,
            quadrature_order=2,
        )
    except ValueError:
        assert True
        return

    assert False, "Expected ValueError for wrong density shape."