from __future__ import annotations

import numpy as np

from magcore.femcore.basis_nedelec import (
    physical_nedelec_basis,
    physical_nedelec_curl,
)
from magcore.femcore.mesh import TetraMesh
from magcore.femcore.quadrature import get_tetra_quadrature
from magcore.femcore.reference_tetra import AffineTetraMap


def local_curlcurl_matrix(
    mesh: TetraMesh,
    cell_idx: int,
    nu: float,
    quadrature_order: int = 1,
) -> np.ndarray:
    """
    Assemble the local curl-curl matrix on one tetrahedron:

        K_ij = ∫_T nu * curl(w_i) · curl(w_j) dV

    For first-order Nedelec elements on an affine tetrahedron, the curls are
    constant on the element, but we still compute via quadrature for consistency.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive.")

    q = get_tetra_quadrature(quadrature_order)
    amap = AffineTetraMap(mesh.cell_vertices(cell_idx))
    detJ = amap.jacobian_determinant()

    K = np.zeros((6, 6), dtype=float)

    curls = [physical_nedelec_curl(amap, i) for i in range(6)]

    for w in q.weights:
        for i in range(6):
            ci = curls[i]
            for j in range(6):
                cj = curls[j]
                K[i, j] += nu * float(np.dot(ci, cj)) * w * detJ

    return K


def local_mass_matrix(
    mesh: TetraMesh,
    cell_idx: int,
    alpha: float,
    quadrature_order: int = 2,
) -> np.ndarray:
    """
    Assemble the local vector mass matrix on one tetrahedron:

        M_ij = ∫_T alpha * w_i · w_j dV
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    q = get_tetra_quadrature(quadrature_order)
    amap = AffineTetraMap(mesh.cell_vertices(cell_idx))
    detJ = amap.jacobian_determinant()

    M = np.zeros((6, 6), dtype=float)

    for xi, w in zip(q.points, q.weights, strict=False):
        x = amap.map_to_physical(xi)
        basis_vals = [physical_nedelec_basis(amap, i, x) for i in range(6)]

        for i in range(6):
            wi = basis_vals[i]
            for j in range(6):
                wj = basis_vals[j]
                M[i, j] += alpha * float(np.dot(wi, wj)) * w * detJ

    return M


def local_rhs_vector(
    mesh: TetraMesh,
    cell_idx: int,
    f_fn,
    quadrature_order: int = 2,
) -> np.ndarray:
    """
    Assemble the local load vector:

        F_i = ∫_T f(x) · w_i(x) dV

    Parameters
    ----------
    f_fn:
        Callable returning a vector of shape (3,) for a physical point x.
    """
    q = get_tetra_quadrature(quadrature_order)
    amap = AffineTetraMap(mesh.cell_vertices(cell_idx))
    detJ = amap.jacobian_determinant()

    F = np.zeros(6, dtype=float)

    for xi, w in zip(q.points, q.weights, strict=False):
        x = amap.map_to_physical(xi)
        f = np.asarray(f_fn(x), dtype=float)

        if f.shape != (3,):
            raise ValueError("f_fn must return vectors of shape (3,).")
        if not np.isfinite(f).all():
            raise ValueError("f_fn must return finite vectors.")

        for i in range(6):
            wi = physical_nedelec_basis(amap, i, x)
            F[i] += float(np.dot(f, wi)) * w * detJ

    return F