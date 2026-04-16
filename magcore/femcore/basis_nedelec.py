from __future__ import annotations

import numpy as np

from magcore.femcore.edge_topology import LOCAL_EDGE_VERTEX_PAIRS
from magcore.femcore.reference_tetra import (
    AffineTetraMap,
    reference_barycentric,
    reference_barycentric_gradients,
)


def reference_nedelec_basis(edge_idx: int, ref_point: np.ndarray) -> np.ndarray:
    """
    First-kind Nedelec basis of order 1 on the reference tetrahedron.

    For local edge e = (i, j):
        w_ij = lambda_i grad(lambda_j) - lambda_j grad(lambda_i)

    Parameters
    ----------
    edge_idx:
        Local edge index in [0, 5]
    ref_point:
        Reference coordinates (x, y, z), shape (3,)
    """
    if not (0 <= edge_idx < 6):
        raise ValueError("edge_idx must be in [0, 5].")

    xi = np.asarray(ref_point, dtype=float)
    if xi.shape != (3,):
        raise ValueError("ref_point must have shape (3,).")

    i, j = LOCAL_EDGE_VERTEX_PAIRS[edge_idx]
    lam = reference_barycentric(xi)
    grads = reference_barycentric_gradients()

    return lam[i] * grads[j] - lam[j] * grads[i]


def reference_nedelec_curl(edge_idx: int) -> np.ndarray:
    """
    Curl of first-kind Nedelec basis on the reference tetrahedron.

    For:
        w_ij = lambda_i grad(lambda_j) - lambda_j grad(lambda_i)

    using constant gradients of barycentric coordinates:
        curl(w_ij) = 2 * grad(lambda_i) x grad(lambda_j)
    """
    if not (0 <= edge_idx < 6):
        raise ValueError("edge_idx must be in [0, 5].")

    i, j = LOCAL_EDGE_VERTEX_PAIRS[edge_idx]
    grads = reference_barycentric_gradients()
    return 2.0 * np.cross(grads[i], grads[j])


def physical_nedelec_basis(
    affine_map: AffineTetraMap,
    edge_idx: int,
    physical_point: np.ndarray,
) -> np.ndarray:
    """
    Covariant Piola transform for H(curl) basis:

        w_phys(x) = J^{-T} w_ref(xi)

    where x = F(xi) = v0 + J xi
    """
    x = np.asarray(physical_point, dtype=float)
    if x.shape != (3,):
        raise ValueError("physical_point must have shape (3,).")

    xi = affine_map.map_to_reference(x)
    w_ref = reference_nedelec_basis(edge_idx, xi)
    return affine_map.inverse_transpose_jacobian() @ w_ref


def physical_nedelec_curl(
    affine_map: AffineTetraMap,
    edge_idx: int,
) -> np.ndarray:
    """
    Curl transform for affine H(curl) basis:

        curl_x(w_phys) = (1 / det(J)) J curl_ref(w_ref)
    """
    curl_ref = reference_nedelec_curl(edge_idx)
    J = affine_map.jacobian_matrix()
    detJ = affine_map.jacobian_determinant()
    return (J @ curl_ref) / detJ


def local_edge_tangent_vector(
    affine_map: AffineTetraMap,
    edge_idx: int,
) -> np.ndarray:
    """
    Physical edge vector corresponding to local edge edge_idx.
    """
    verts = affine_map.physical_vertices
    i, j = LOCAL_EDGE_VERTEX_PAIRS[edge_idx]
    return verts[j] - verts[i]


def edge_line_integral_of_basis_on_straight_edge(
    affine_map: AffineTetraMap,
    basis_edge_idx: int,
    test_edge_idx: int,
    n_points: int = 8,
) -> float:
    """
    Numerically evaluate the line integral of one physical Nedelec basis function
    along a physical local edge.

    Since the edge is straight, parameterize:
        x(t) = v_i + t (v_j - v_i),  t in [0, 1]

    and compute:
        ∫ w_phys(x(t)) · x'(t) dt
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    verts = affine_map.physical_vertices
    i, j = LOCAL_EDGE_VERTEX_PAIRS[test_edge_idx]
    vi = verts[i]
    vj = verts[j]
    tau = vj - vi

    ts = np.linspace(0.0, 1.0, n_points, dtype=float)
    vals = []

    for t in ts:
        x = vi + t * tau
        w = physical_nedelec_basis(affine_map, basis_edge_idx, x)
        vals.append(np.dot(w, tau))

    vals = np.asarray(vals, dtype=float)
    return float(np.trapezoid(vals, ts))