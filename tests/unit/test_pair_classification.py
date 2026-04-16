from __future__ import annotations

import numpy as np

from magcore.bem.pair_classification import (
    FacePairRelation,
    face_pair_relation,
    shared_vertices,
)
from magcore.mesh.surface_mesh import SurfaceMesh


def make_relation_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # 0
            [1.0, 0.0, 0.0],   # 1
            [0.0, 1.0, 0.0],   # 2
            [1.0, 1.0, 0.0],   # 3
            [2.0, 0.0, 0.0],   # 4
            [5.0, 0.0, 0.0],   # 5
            [6.0, 0.0, 0.0],   # 6
            [5.0, 1.0, 0.0],   # 7
            [0.2, 0.2, 0.2],   # 8
            [1.2, 0.2, 0.2],   # 9
            [0.2, 1.2, 0.2],   # 10
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],   # face 0
            [1, 3, 2],   # face 1: shared edge with face 0
            [1, 4, 3],   # face 2: shared vertex with face 0
            [5, 6, 7],   # face 3: far regular face
            [8, 9, 10],  # face 4: near but disjoint from face 0
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_shared_vertices_for_shared_edge_pair():
    mesh = make_relation_mesh()
    s = shared_vertices(mesh, 0, 1)

    assert s == (1, 2)


def test_shared_vertices_for_shared_vertex_pair():
    mesh = make_relation_mesh()
    s = shared_vertices(mesh, 0, 2)

    assert s == (1,)


def test_face_pair_relation_self():
    mesh = make_relation_mesh()
    rel = face_pair_relation(mesh, 0, 0)

    assert rel == FacePairRelation.SELF


def test_face_pair_relation_shared_edge():
    mesh = make_relation_mesh()
    rel = face_pair_relation(mesh, 0, 1)

    assert rel == FacePairRelation.SHARED_EDGE


def test_face_pair_relation_shared_vertex():
    mesh = make_relation_mesh()
    rel = face_pair_relation(mesh, 0, 2)

    assert rel == FacePairRelation.SHARED_VERTEX


def test_face_pair_relation_regular():
    mesh = make_relation_mesh()
    rel = face_pair_relation(mesh, 0, 3, near_factor=1.0)

    assert rel == FacePairRelation.REGULAR


def test_face_pair_relation_near():
    mesh = make_relation_mesh()
    rel = face_pair_relation(mesh, 0, 4, near_factor=10.0)

    assert rel == FacePairRelation.NEAR