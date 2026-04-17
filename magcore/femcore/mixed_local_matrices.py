from __future__ import annotations

import numpy as np

from magcore.femcore.basis_nedelec import physical_nedelec_basis
from magcore.femcore.local_matrices import local_curlcurl_matrix
from magcore.femcore.quadrature import get_tetra_quadrature
from magcore.femcore.reference_tetra import AffineTetraMap
from magcore.mesh.mesh import TetraMesh


def local_vector_source_rhs(
    mesh: TetraMesh,
    cell_idx: int,
    J_fn,
    quadrature_order: int = 3,
) -> np.ndarray:
    q = get_tetra_quadrature(quadrature_order)
    amap = AffineTetraMap(mesh.cell_vertices(cell_idx))
    detJ = amap.jacobian_determinant()

    F = np.zeros(6, dtype=float)

    for xi, w in zip(q.points, q.weights, strict=False):
        x = amap.map_to_physical(xi)
        J = np.asarray(J_fn(x), dtype=float)

        if J.shape != (3,):
            raise ValueError("J_fn must return vectors of shape (3,).")
        if not np.isfinite(J).all():
            raise ValueError("J_fn must return finite vectors.")

        for i in range(6):
            wi = physical_nedelec_basis(amap, i, x)
            F[i] += float(np.dot(J, wi)) * w * detJ

    return F


def local_grad_p_coupling_matrix(
    mesh: TetraMesh,
    cell_idx: int,
    quadrature_order: int = 2,
) -> np.ndarray:
    q = get_tetra_quadrature(quadrature_order)
    amap = AffineTetraMap(mesh.cell_vertices(cell_idx))
    detJ = amap.jacobian_determinant()
    grads_phi = amap.physical_barycentric_gradients()

    G = np.zeros((6, 4), dtype=float)

    for xi, w in zip(q.points, q.weights, strict=False):
        x = amap.map_to_physical(xi)
        basis_vals = [physical_nedelec_basis(amap, i, x) for i in range(6)]

        for i in range(6):
            wi = basis_vals[i]
            for j in range(4):
                grad_phi_j = grads_phi[j]
                G[i, j] += float(np.dot(grad_phi_j, wi)) * w * detJ

    return G


def local_scalar_mass_matrix(
    mesh: TetraMesh,
    cell_idx: int,
    beta: float = 1.0,
    quadrature_order: int = 2,
) -> np.ndarray:
    if beta <= 0.0:
        raise ValueError("beta must be positive.")

    q = get_tetra_quadrature(quadrature_order)
    amap = AffineTetraMap(mesh.cell_vertices(cell_idx))
    detJ = amap.jacobian_determinant()

    M = np.zeros((4, 4), dtype=float)

    for xi, w in zip(q.points, q.weights, strict=False):
        lam = amap.barycentric_at_physical_point(amap.map_to_physical(xi))
        for i in range(4):
            for j in range(4):
                M[i, j] += beta * lam[i] * lam[j] * w * detJ

    return M


def local_curlcurl_block(
    mesh: TetraMesh,
    cell_idx: int,
    nu: float,
    quadrature_order: int = 1,
) -> np.ndarray:
    return local_curlcurl_matrix(
        mesh=mesh,
        cell_idx=cell_idx,
        nu=nu,
        quadrature_order=quadrature_order,
    )