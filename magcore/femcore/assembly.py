from __future__ import annotations

import numpy as np

from magcore.femcore.local_matrices import (
    local_curlcurl_matrix,
    local_mass_matrix,
    local_rhs_vector,
)
from magcore.femcore.mesh import TetraMesh
from magcore.femcore.spaces import NedelecP1Space


def assemble_curlcurl_matrix(
    mesh: TetraMesh,
    space: NedelecP1Space,
    nu: float,
    quadrature_order: int = 1,
) -> np.ndarray:
    ndofs = space.ndofs
    A = np.zeros((ndofs, ndofs), dtype=float)

    for cell_idx in range(mesh.n_cells):
        Ke = local_curlcurl_matrix(
            mesh=mesh,
            cell_idx=cell_idx,
            nu=nu,
            quadrature_order=quadrature_order,
        )
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


def assemble_mass_matrix(
    mesh: TetraMesh,
    space: NedelecP1Space,
    alpha: float,
    quadrature_order: int = 2,
) -> np.ndarray:
    ndofs = space.ndofs
    M = np.zeros((ndofs, ndofs), dtype=float)

    for cell_idx in range(mesh.n_cells):
        Me = local_mass_matrix(
            mesh=mesh,
            cell_idx=cell_idx,
            alpha=alpha,
            quadrature_order=quadrature_order,
        )
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


def assemble_rhs_vector(
    mesh: TetraMesh,
    space: NedelecP1Space,
    f_fn,
    quadrature_order: int = 2,
) -> np.ndarray:
    ndofs = space.ndofs
    b = np.zeros(ndofs, dtype=float)

    for cell_idx in range(mesh.n_cells):
        fe = local_rhs_vector(
            mesh=mesh,
            cell_idx=cell_idx,
            f_fn=f_fn,
            quadrature_order=quadrature_order,
        )
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
    """
    Assemble the global linear system:

        A = K + M
        b = F

    with:
        K_ij = ∫ nu * curl(w_i)·curl(w_j)
        M_ij = ∫ alpha * w_i·w_j
        F_i  = ∫ f·w_i
    """
    K = assemble_curlcurl_matrix(
        mesh=mesh,
        space=space,
        nu=nu,
        quadrature_order=curl_quadrature_order,
    )
    M = assemble_mass_matrix(
        mesh=mesh,
        space=space,
        alpha=alpha,
        quadrature_order=mass_quadrature_order,
    )
    b = assemble_rhs_vector(
        mesh=mesh,
        space=space,
        f_fn=f_fn,
        quadrature_order=rhs_quadrature_order,
    )

    return K + M, b