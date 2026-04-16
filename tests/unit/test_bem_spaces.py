from __future__ import annotations

import numpy as np

from magcore.mesh.surface_mesh import SurfaceMesh
from magcore.bem.spaces import FaceP0Space, VertexP1Space
from magcore.enums import BasisKind


def make_two_triangle_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
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


def test_face_p0_space_on_all_faces():
    mesh = make_two_triangle_mesh()
    space = FaceP0Space.from_faces(mesh, (0, 1))

    assert space.basis_kind == BasisKind.P0
    assert space.ndofs == 2
    assert space.dof_to_face == (0, 1)
    assert space.face_to_dof[0] == 0
    assert space.face_to_dof[1] == 1


def test_vertex_p1_space_on_all_faces():
    mesh = make_two_triangle_mesh()
    space = VertexP1Space.from_faces(mesh, (0, 1))

    assert space.basis_kind == BasisKind.P1
    assert space.ndofs == 4
    assert space.dof_to_vertex == (0, 1, 2, 3)
    assert space.vertex_to_dof[0] == 0
    assert space.vertex_to_dof[3] == 3


def test_vertex_p1_space_on_subset():
    mesh = make_two_triangle_mesh()
    space = VertexP1Space.from_faces(mesh, (0,))

    assert space.ndofs == 3
    assert set(space.active_vertices) == {0, 1, 2}


def test_face_p0_space_on_subset():
    mesh = make_two_triangle_mesh()
    space = FaceP0Space.from_faces(mesh, (1,))

    assert space.ndofs == 1
    assert space.dof_to_face == (1,)
    assert space.face_to_dof[1] == 0