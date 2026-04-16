from __future__ import annotations

import numpy as np

from magcore.bem.element_integrals import triangle_area, triangle_centroid, triangle_diameter


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Midpoint of two 3D points.
    """
    return 0.5 * (a + b)


def subdivide_triangle_4(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split one physical triangle into 4 subtriangles by edge midpoints.

    Vertex convention:
        tri = [v0, v1, v2]

    Midpoints:
        m01 = (v0+v1)/2
        m12 = (v1+v2)/2
        m20 = (v2+v0)/2

    Returns four triangles:
        [v0,  m01, m20]
        [m01, v1,  m12]
        [m20, m12, v2 ]
        [m01, m12, m20]
    """
    tri = np.asarray(tri, dtype=float)
    if tri.shape != (3, 3):
        raise ValueError("tri must have shape (3, 3).")

    v0, v1, v2 = tri
    m01 = midpoint(v0, v1)
    m12 = midpoint(v1, v2)
    m20 = midpoint(v2, v0)

    t0 = np.array([v0, m01, m20], dtype=float)
    t1 = np.array([m01, v1, m12], dtype=float)
    t2 = np.array([m20, m12, v2], dtype=float)
    t3 = np.array([m01, m12, m20], dtype=float)

    return t0, t1, t2, t3


def subdivide_mesh_face_triangle(mesh, face_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a triangle from mesh by face index and subdivide it into 4 subtriangles.
    """
    tri = mesh.face_vertices(face_idx)
    return subdivide_triangle_4(tri)


def triangle_pair_distance_proxy(tri_a: np.ndarray, tri_b: np.ndarray) -> float:
    """
    Cheap geometric proxy for pair distance:
    centroid-to-centroid distance.
    """
    ca = triangle_centroid(tri_a)
    cb = triangle_centroid(tri_b)
    return float(np.linalg.norm(ca - cb))


def triangle_pair_is_regular(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
    near_factor: float = 2.0,
) -> bool:
    """
    Local criterion for whether a pair of physical triangles can be treated
    by regular quadrature at the current subdivision level.

    Uses:
        dist(centroids) >= near_factor * max(diameters)
    """
    dist = triangle_pair_distance_proxy(tri_a, tri_b)
    da = triangle_diameter(tri_a)
    db = triangle_diameter(tri_b)
    return dist >= near_factor * max(da, db)


def subdivision_area_conservation_error(tri: np.ndarray) -> float:
    """
    Diagnostic helper:
    absolute area mismatch between parent triangle and sum of 4 children.
    """
    parent_area = triangle_area(tri)
    children = subdivide_triangle_4(tri)
    child_area = sum(triangle_area(t) for t in children)
    return abs(parent_area - child_area)