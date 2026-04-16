from __future__ import annotations

import numpy as np

from magcore.bem.closed_surface_checks import check_closed_face_set
from magcore.mesh.surface_mesh import SurfaceMesh


def make_tetra_surface_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def make_open_mesh() -> SurfaceMesh:
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


def test_check_closed_face_set_detects_closed_tetra_surface():
    mesh = make_tetra_surface_mesh()
    report = check_closed_face_set(mesh, (0, 1, 2, 3))

    assert report.is_closed is True
    assert report.n_boundary_edges == 0
    assert report.n_non_manifold_edges == 0


def test_check_closed_face_set_detects_open_surface():
    mesh = make_open_mesh()
    report = check_closed_face_set(mesh, (0,))

    assert report.is_closed is False
    assert report.n_boundary_edges > 0