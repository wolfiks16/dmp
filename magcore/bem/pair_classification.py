from __future__ import annotations

from enum import Enum

import numpy as np

from magcore.bem.element_integrals import (
    triangle_centroid,
    triangle_diameter,
    triangle_vertices,
)
from magcore.mesh.surface_mesh import SurfaceMesh


class FacePairRelation(str, Enum):
    SELF = "self"
    SHARED_EDGE = "shared_edge"
    SHARED_VERTEX = "shared_vertex"
    NEAR = "near"
    REGULAR = "regular"


def shared_vertices(mesh: SurfaceMesh, face_i: int, face_j: int) -> tuple[int, ...]:
    """
    Return sorted shared vertex indices between two mesh faces.
    """
    vi = set(int(v) for v in mesh.faces[face_i])
    vj = set(int(v) for v in mesh.faces[face_j])
    return tuple(sorted(vi.intersection(vj)))


def face_pair_relation(
    mesh: SurfaceMesh,
    face_i: int,
    face_j: int,
    near_factor: float = 2.0,
) -> FacePairRelation:
    """
    Classify the geometric relation between two mesh faces.

    Rules:
    - same face -> SELF
    - 2 shared vertices -> SHARED_EDGE
    - 1 shared vertex -> SHARED_VERTEX
    - 0 shared vertices but close by centroid-diameter test -> NEAR
    - otherwise -> REGULAR

    The near criterion is intentionally conservative.
    """
    if face_i == face_j:
        return FacePairRelation.SELF

    sverts = shared_vertices(mesh, face_i, face_j)
    n_shared = len(sverts)

    if n_shared >= 2:
        return FacePairRelation.SHARED_EDGE
    if n_shared == 1:
        return FacePairRelation.SHARED_VERTEX

    tri_i = triangle_vertices(mesh, face_i)
    tri_j = triangle_vertices(mesh, face_j)

    ci = triangle_centroid(tri_i)
    cj = triangle_centroid(tri_j)
    dist = float(np.linalg.norm(ci - cj))

    di = triangle_diameter(tri_i)
    dj = triangle_diameter(tri_j)

    if dist < near_factor * max(di, dj):
        return FacePairRelation.NEAR

    return FacePairRelation.REGULAR