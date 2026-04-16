from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh


def make_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_surface_mesh_basic_counts():
    mesh = make_mesh()
    assert mesh.n_vertices == 3
    assert mesh.n_faces == 1


def test_face_vertices_returns_triangle_vertices():
    mesh = make_mesh()
    tri = mesh.face_vertices(0)

    assert tri.shape == (3, 3)
    assert np.allclose(tri[0], [0.0, 0.0, 0.0])
    assert np.allclose(tri[1], [2.0, 0.0, 0.0])
    assert np.allclose(tri[2], [0.0, 2.0, 0.0])


def test_face_area_is_correct():
    mesh = make_mesh()
    area = mesh.face_area(0)

    assert np.isclose(area, 2.0)


def test_face_centroid_is_correct():
    mesh = make_mesh()
    centroid = mesh.face_centroid(0)

    assert np.allclose(centroid, [2.0 / 3.0, 2.0 / 3.0, 0.0])


def test_face_normal_is_unit_and_oriented():
    mesh = make_mesh()
    normal = mesh.face_normal(0)

    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.allclose(normal, [0.0, 0.0, 1.0])


def test_validate_basic_accepts_valid_mesh():
    mesh = make_mesh()
    issues = mesh.validate_basic()

    assert issues == ()


def test_validate_basic_detects_degenerate_face():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = SurfaceMesh(vertices=vertices, faces=faces)

    issues = mesh.validate_basic()
    codes = {i.code for i in issues}

    assert "mesh.face.degenerate" in codes