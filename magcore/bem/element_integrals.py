from __future__ import annotations

from enum import Enum
from typing import Callable

import numpy as np

from magcore.bem.laplace_kernels import laplace_green_3d
from magcore.bem.quadrature import QuadratureRule
from magcore.mesh.surface_mesh import SurfaceMesh


class FacePairClass(str, Enum):
    REGULAR = "regular"
    NEAR = "near"
    SINGULAR = "singular"


def triangle_vertices(mesh: SurfaceMesh, face_idx: int) -> np.ndarray:
    """
    Return the 3x3 array of triangle vertex coordinates for a mesh face.
    """
    return mesh.face_vertices(face_idx)


def triangle_area(tri: np.ndarray) -> float:
    """
    Area of a physical triangle given as shape (3, 3).
    """
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    return 0.5 * float(np.linalg.norm(np.cross(e1, e2)))


def triangle_jacobian_scale(tri: np.ndarray) -> float:
    """
    Surface Jacobian scale from the reference triangle to the physical triangle.

    For an affine triangle map:
        y = v0 + xi*(v1-v0) + eta*(v2-v0)

    dS = || (v1-v0) x (v2-v0) || dxi deta
    so this value equals 2 * area(tri).
    """
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    return float(np.linalg.norm(np.cross(e1, e2)))


def map_reference_triangle_to_physical(tri: np.ndarray, xi_eta: np.ndarray) -> np.ndarray:
    """
    Map a reference triangle point (xi, eta) to the physical triangle.

    Reference triangle:
        xi >= 0, eta >= 0, xi + eta <= 1
    """
    xi, eta = float(xi_eta[0]), float(xi_eta[1])
    v0, v1, v2 = tri
    return v0 + xi * (v1 - v0) + eta * (v2 - v0)


def triangle_centroid(tri: np.ndarray) -> np.ndarray:
    """
    Centroid of a physical triangle.
    """
    return tri.mean(axis=0)


def triangle_diameter(tri: np.ndarray) -> float:
    """
    Maximum edge length of a triangle.
    """
    d01 = np.linalg.norm(tri[1] - tri[0])
    d12 = np.linalg.norm(tri[2] - tri[1])
    d20 = np.linalg.norm(tri[0] - tri[2])
    return float(max(d01, d12, d20))


def faces_share_vertex(mesh: SurfaceMesh, face_i: int, face_j: int) -> bool:
    """
    Conservative singularity detector: if two faces share a vertex,
    treat the pair as singular at this stage.
    """
    vi = set(int(v) for v in mesh.faces[face_i])
    vj = set(int(v) for v in mesh.faces[face_j])
    return len(vi.intersection(vj)) > 0


def classify_face_pair(
    mesh: SurfaceMesh,
    face_i: int,
    face_j: int,
    near_factor: float = 2.0,
) -> FacePairClass:
    """
    Classify a pair of faces for regular-only assembly.

    Policy in R1.4:
    - same face -> SINGULAR
    - shared vertex -> SINGULAR
    - close by centroid-distance criterion -> NEAR
    - otherwise -> REGULAR

    This policy is intentionally conservative.
    """
    if face_i == face_j:
        return FacePairClass.SINGULAR

    if faces_share_vertex(mesh, face_i, face_j):
        return FacePairClass.SINGULAR

    tri_i = triangle_vertices(mesh, face_i)
    tri_j = triangle_vertices(mesh, face_j)

    ci = triangle_centroid(tri_i)
    cj = triangle_centroid(tri_j)
    dist = float(np.linalg.norm(ci - cj))

    di = triangle_diameter(tri_i)
    dj = triangle_diameter(tri_j)

    if dist < near_factor * max(di, dj):
        return FacePairClass.NEAR

    return FacePairClass.REGULAR


def integrate_over_triangle_regular(
    tri: np.ndarray,
    integrand: Callable[[np.ndarray], float],
    quadrature: QuadratureRule,
) -> float:
    """
    Regular numerical integration over a physical triangle.

    Computes:
        ∫_T f(y) dS_y

    using a quadrature rule defined on the reference triangle.
    """
    jac = triangle_jacobian_scale(tri)
    acc = 0.0

    for qp, w in zip(quadrature.points, quadrature.weights, strict=False):
        y = map_reference_triangle_to_physical(tri, qp)
        acc += float(w) * float(integrand(y))

    return jac * acc


def single_layer_point_potential_regular(
    target_point: np.ndarray,
    source_tri: np.ndarray,
    quadrature: QuadratureRule,
) -> float:
    """
    Compute the regular single-layer potential contribution of one source triangle
    at one target point:

        ∫_T G(x, y) dS_y

    This routine assumes the target point is not on or too close to the source
    triangle in a singular or near-singular sense.
    """
    return integrate_over_triangle_regular(
        source_tri,
        lambda y: laplace_green_3d(target_point, y),
        quadrature,
    )


def single_layer_face_face_regular(
    target_tri: np.ndarray,
    source_tri: np.ndarray,
    target_quadrature: QuadratureRule,
    source_quadrature: QuadratureRule,
) -> float:
    """
    Compute the regular P0-P0 single-layer interaction between two disjoint,
    sufficiently separated triangles:

        ∫_{T_i} ∫_{T_j} G(x, y) dS_y dS_x
    """
    jac_t = triangle_jacobian_scale(target_tri)
    jac_s = triangle_jacobian_scale(source_tri)

    acc = 0.0
    for qp_t, wt in zip(target_quadrature.points, target_quadrature.weights, strict=False):
        x = map_reference_triangle_to_physical(target_tri, qp_t)
        for qp_s, ws in zip(source_quadrature.points, source_quadrature.weights, strict=False):
            y = map_reference_triangle_to_physical(source_tri, qp_s)
            acc += float(wt) * float(ws) * laplace_green_3d(x, y)

    return jac_t * jac_s * acc