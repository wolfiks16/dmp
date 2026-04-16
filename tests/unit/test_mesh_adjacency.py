from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.mesh.adjacency import (
    build_edge_to_faces,
    build_face_to_faces,
    find_boundary_edges,
    find_non_manifold_edges,
    patch_connected_components,
)


def make_two_triangle_square() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_build_edge_to_faces_detects_shared_diagonal():
    mesh = make_two_triangle_square()
    edge_to_faces = build_edge_to_faces(mesh)

    assert edge_to_faces[(1, 2)] == (0, 1)


def test_build_face_to_faces_detects_neighbors():
    mesh = make_two_triangle_square()
    face_to_faces = build_face_to_faces(mesh)

    assert face_to_faces[0] == (1,)
    assert face_to_faces[1] == (0,)


def test_find_boundary_edges_on_open_two_triangle_patch():
    mesh = make_two_triangle_square()
    boundary = set(find_boundary_edges(mesh))

    assert boundary == {(0, 1), (0, 2), (1, 3), (2, 3)}


def test_find_non_manifold_edges_empty_for_two_triangles():
    mesh = make_two_triangle_square()
    non_manifold = find_non_manifold_edges(mesh)

    assert non_manifold == {}


def test_patch_connected_components_single_component():
    mesh = make_two_triangle_square()
    comps = patch_connected_components(mesh, (0, 1))

    assert comps == ((0, 1),)


def test_patch_connected_components_multiple_components():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
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
    mesh = SurfaceMesh(vertices=vertices, faces=faces)

    comps = patch_connected_components(mesh, (0, 1))
    assert len(comps) == 2