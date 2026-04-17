from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_mass_matrix
from magcore.femcore.mesh import TetraMesh
from magcore.femcore.quadrature import get_tetra_quadrature
from magcore.femcore.reference_tetra import AffineTetraMap
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space


def compute_gauge_residual_vector(G: np.ndarray, a: np.ndarray) -> np.ndarray:
    G = np.asarray(G, dtype=float)
    a = np.asarray(a, dtype=float)
    if G.ndim != 2:
        raise ValueError("G must be a 2D matrix.")
    if a.ndim != 1 or a.shape[0] != G.shape[0]:
        raise ValueError("a must have shape (G.shape[0],).")
    return G.T @ a


def compute_gauge_residual_norm(G: np.ndarray, a: np.ndarray) -> float:
    return float(np.linalg.norm(compute_gauge_residual_vector(G, a)))


def assemble_scalar_stiffness_matrix_p1(mesh: TetraMesh, scalar_space: LagrangeP1Space) -> np.ndarray:
    if scalar_space.mesh is not mesh:
        raise ValueError("scalar_space must be built on the provided mesh.")
    S = np.zeros((scalar_space.ndofs, scalar_space.ndofs), dtype=float)
    for cell_idx in range(mesh.n_cells):
        amap = AffineTetraMap(mesh.cell_vertices(cell_idx))
        detJ = amap.jacobian_determinant()
        grads = amap.physical_barycentric_gradients()
        vol = detJ / 6.0
        gdofs = scalar_space.cell_dof_indices(cell_idx)
        for i in range(4):
            gi = gdofs[i]
            for j in range(4):
                gj = gdofs[j]
                S[gi, gj] += float(np.dot(grads[i], grads[j])) * vol
    return S


def project_to_scalar_gradient_subspace(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    a: np.ndarray,
    G: np.ndarray,
) -> tuple[np.ndarray, float]:
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.shape[0] != vector_space.ndofs:
        raise ValueError("a must have shape (vector_space.ndofs,).")
    if G.shape != (vector_space.ndofs, scalar_space.ndofs):
        raise ValueError("G must have shape (vector_space.ndofs, scalar_space.ndofs).")

    S = assemble_scalar_stiffness_matrix_p1(mesh, scalar_space)
    rhs = G.T @ a
    boundary = np.asarray(scalar_space.boundary_dofs(), dtype=int)
    free = np.setdiff1d(np.arange(scalar_space.ndofs, dtype=int), boundary)

    phi = np.zeros(scalar_space.ndofs, dtype=float)
    if free.size > 0:
        phi_free = np.linalg.solve(S[np.ix_(free, free)], rhs[free])
        phi[free] = phi_free
    longitudinal_norm_sq = float(phi @ (S @ phi))
    if longitudinal_norm_sq < 0.0 and abs(longitudinal_norm_sq) < 1e-12:
        longitudinal_norm_sq = 0.0
    longitudinal_norm = float(np.sqrt(longitudinal_norm_sq))
    return phi, longitudinal_norm


def compute_relative_longitudinal_component(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    a: np.ndarray,
    G: np.ndarray,
    vector_mass_alpha: float = 1.0,
    vector_mass_quadrature_order: int = 2,
) -> float:
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.shape[0] != vector_space.ndofs:
        raise ValueError("a must have shape (vector_space.ndofs,).")
    M = assemble_mass_matrix(mesh=mesh, space=vector_space, alpha=vector_mass_alpha, quadrature_order=vector_mass_quadrature_order)
    total_norm_sq = float(a @ (M @ a))
    if total_norm_sq <= 0.0:
        if abs(total_norm_sq) < 1e-14:
            return 0.0
        raise ValueError("Vector potential norm must be positive.")
    _, longitudinal_norm = project_to_scalar_gradient_subspace(mesh, vector_space, scalar_space, a, G)
    return float(longitudinal_norm / np.sqrt(total_norm_sq))
