from __future__ import annotations

import numpy as np

from magcore.femcore.local_matrices import local_curlcurl_matrix, local_mass_matrix, local_rhs_vector
from magcore.femcore.mesh import TetraMesh
from magcore.femcore.mixed_local_matrices import (
    local_curlcurl_block,
    local_grad_p_coupling_matrix,
    local_scalar_mass_matrix,
    local_vector_source_rhs,
)
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space


def assemble_curlcurl_matrix(mesh: TetraMesh, space: NedelecP1Space, nu: float, quadrature_order: int = 1) -> np.ndarray:
    ndofs = space.ndofs
    A = np.zeros((ndofs, ndofs), dtype=float)
    for cell_idx in range(mesh.n_cells):
        Ke = local_curlcurl_matrix(mesh=mesh, cell_idx=cell_idx, nu=nu, quadrature_order=quadrature_order)
        gdofs = space.cell_dof_indices(cell_idx)
        sgn = space.cell_dof_signs(cell_idx)
        for i in range(6):
            gi = gdofs[i]
            si = sgn[i]
            for j in range(6):
                gj = gdofs[j]
                sj = sgn[j]
                A[gi, gj] += si * sj * Ke[i, j]
    return A


def assemble_mass_matrix(mesh: TetraMesh, space: NedelecP1Space, alpha: float, quadrature_order: int = 2) -> np.ndarray:
    ndofs = space.ndofs
    M = np.zeros((ndofs, ndofs), dtype=float)
    for cell_idx in range(mesh.n_cells):
        Me = local_mass_matrix(mesh=mesh, cell_idx=cell_idx, alpha=alpha, quadrature_order=quadrature_order)
        gdofs = space.cell_dof_indices(cell_idx)
        sgn = space.cell_dof_signs(cell_idx)
        for i in range(6):
            gi = gdofs[i]
            si = sgn[i]
            for j in range(6):
                gj = gdofs[j]
                sj = sgn[j]
                M[gi, gj] += si * sj * Me[i, j]
    return M


def assemble_rhs_vector(mesh: TetraMesh, space: NedelecP1Space, f_fn, quadrature_order: int = 2) -> np.ndarray:
    ndofs = space.ndofs
    b = np.zeros(ndofs, dtype=float)
    for cell_idx in range(mesh.n_cells):
        fe = local_rhs_vector(mesh=mesh, cell_idx=cell_idx, f_fn=f_fn, quadrature_order=quadrature_order)
        gdofs = space.cell_dof_indices(cell_idx)
        sgn = space.cell_dof_signs(cell_idx)
        for i in range(6):
            gi = gdofs[i]
            si = sgn[i]
            b[gi] += si * fe[i]
    return b


def assemble_system(
    mesh: TetraMesh,
    space: NedelecP1Space,
    nu: float,
    alpha: float,
    f_fn,
    curl_quadrature_order: int = 1,
    mass_quadrature_order: int = 2,
    rhs_quadrature_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    K = assemble_curlcurl_matrix(mesh=mesh, space=space, nu=nu, quadrature_order=curl_quadrature_order)
    M = assemble_mass_matrix(mesh=mesh, space=space, alpha=alpha, quadrature_order=mass_quadrature_order)
    b = assemble_rhs_vector(mesh=mesh, space=space, f_fn=f_fn, quadrature_order=rhs_quadrature_order)
    return K + M, b


def _validate_mixed_spaces(mesh: TetraMesh, vector_space: NedelecP1Space, scalar_space: LagrangeP1Space) -> None:
    if vector_space.mesh is not mesh:
        raise ValueError("vector_space must be built on the provided mesh.")
    if scalar_space.mesh is not mesh:
        raise ValueError("scalar_space must be built on the provided mesh.")


def assemble_mixed_curlcurl_block(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    nu: float,
    quadrature_order: int = 1,
) -> np.ndarray:
    _validate_mixed_spaces(mesh, vector_space, LagrangeP1Space(mesh))
    ndofs = vector_space.ndofs
    K = np.zeros((ndofs, ndofs), dtype=float)
    for cell_idx in range(mesh.n_cells):
        Ke = local_curlcurl_block(mesh=mesh, cell_idx=cell_idx, nu=nu, quadrature_order=quadrature_order)
        gdofs = vector_space.cell_dof_indices(cell_idx)
        sgn = vector_space.cell_dof_signs(cell_idx)
        for i in range(6):
            gi = gdofs[i]
            si = sgn[i]
            for j in range(6):
                gj = gdofs[j]
                sj = sgn[j]
                K[gi, gj] += si * sj * Ke[i, j]
    return K


def assemble_mixed_grad_p_coupling_matrix(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    quadrature_order: int = 2,
) -> np.ndarray:
    _validate_mixed_spaces(mesh, vector_space, scalar_space)
    G = np.zeros((vector_space.ndofs, scalar_space.ndofs), dtype=float)
    for cell_idx in range(mesh.n_cells):
        Ge = local_grad_p_coupling_matrix(mesh=mesh, cell_idx=cell_idx, quadrature_order=quadrature_order)
        vg = vector_space.cell_dof_indices(cell_idx)
        vs = vector_space.cell_dof_signs(cell_idx)
        pg = scalar_space.cell_dof_indices(cell_idx)
        for i in range(6):
            gi = vg[i]
            si = vs[i]
            for j in range(4):
                gj = pg[j]
                G[gi, gj] += si * Ge[i, j]
    return G


def assemble_mixed_vector_source_rhs(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    J_fn,
    quadrature_order: int = 2,
) -> np.ndarray:
    b = np.zeros(vector_space.ndofs, dtype=float)
    for cell_idx in range(mesh.n_cells):
        Fe = local_vector_source_rhs(mesh=mesh, cell_idx=cell_idx, J_fn=J_fn, quadrature_order=quadrature_order)
        gdofs = vector_space.cell_dof_indices(cell_idx)
        sgn = vector_space.cell_dof_signs(cell_idx)
        for i in range(6):
            gi = gdofs[i]
            si = sgn[i]
            b[gi] += si * Fe[i]
    return b


def assemble_scalar_mass_matrix_p1(
    mesh: TetraMesh,
    scalar_space: LagrangeP1Space,
    beta: float = 1.0,
    quadrature_order: int = 2,
) -> np.ndarray:
    M = np.zeros((scalar_space.ndofs, scalar_space.ndofs), dtype=float)
    for cell_idx in range(mesh.n_cells):
        Me = local_scalar_mass_matrix(mesh=mesh, cell_idx=cell_idx, beta=beta, quadrature_order=quadrature_order)
        gdofs = scalar_space.cell_dof_indices(cell_idx)
        for i in range(4):
            gi = gdofs[i]
            for j in range(4):
                gj = gdofs[j]
                M[gi, gj] += Me[i, j]
    return M


def assemble_mixed_coulomb_system(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    nu: float,
    J_fn,
    curl_quadrature_order: int = 1,
    coupling_quadrature_order: int = 2,
    rhs_quadrature_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_mixed_spaces(mesh, vector_space, scalar_space)
    K = np.zeros((vector_space.ndofs, vector_space.ndofs), dtype=float)
    G = np.zeros((vector_space.ndofs, scalar_space.ndofs), dtype=float)
    f = np.zeros(vector_space.ndofs, dtype=float)

    for cell_idx in range(mesh.n_cells):
        Ke = local_curlcurl_block(mesh=mesh, cell_idx=cell_idx, nu=nu, quadrature_order=curl_quadrature_order)
        Ge = local_grad_p_coupling_matrix(mesh=mesh, cell_idx=cell_idx, quadrature_order=coupling_quadrature_order)
        Fe = local_vector_source_rhs(mesh=mesh, cell_idx=cell_idx, J_fn=J_fn, quadrature_order=rhs_quadrature_order)

        vg = vector_space.cell_dof_indices(cell_idx)
        vs = vector_space.cell_dof_signs(cell_idx)
        pg = scalar_space.cell_dof_indices(cell_idx)

        for i in range(6):
            gi = vg[i]
            si = vs[i]
            f[gi] += si * Fe[i]
            for j in range(6):
                gj = vg[j]
                sj = vs[j]
                K[gi, gj] += si * sj * Ke[i, j]
            for j in range(4):
                gj = pg[j]
                G[gi, gj] += si * Ge[i, j]

    n_a = vector_space.ndofs
    n_p = scalar_space.ndofs
    A = np.zeros((n_a + n_p, n_a + n_p), dtype=float)
    rhs = np.zeros(n_a + n_p, dtype=float)
    A[:n_a, :n_a] = K
    A[:n_a, n_a:] = G
    A[n_a:, :n_a] = G.T
    rhs[:n_a] = f
    return A, rhs, K, G
