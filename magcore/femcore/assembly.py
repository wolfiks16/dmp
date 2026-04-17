from __future__ import annotations

import numpy as np

from magcore.femcore.local_matrices import (
    local_curlcurl_matrix,
    local_mass_matrix,
    local_rhs_vector,
)
from magcore.femcore.mixed_local_matrices import (
    local_curlcurl_block,
    local_grad_p_coupling_matrix,
    local_vector_source_rhs,
)
from magcore.femcore.quadrature import get_tetra_quadrature
from magcore.femcore.reference_tetra import AffineTetraMap
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh import TetraMesh


def _validate_spaces_same_mesh(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space | None = None,
) -> None:
    if vector_space.mesh is not mesh:
        raise ValueError("vector_space must be built on the provided mesh.")
    if scalar_space is not None and scalar_space.mesh is not mesh:
        raise ValueError("scalar_space must be built on the provided mesh.")


def assemble_curlcurl_matrix(
    mesh: TetraMesh,
    space: NedelecP1Space,
    nu: float,
    quadrature_order: int = 1,
) -> np.ndarray:
    _validate_spaces_same_mesh(mesh, space)

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
    _validate_spaces_same_mesh(mesh, space)

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
    quadrature_order: int = 3,
) -> np.ndarray:
    _validate_spaces_same_mesh(mesh, space)

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
    rhs_quadrature_order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
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


def assemble_vector_source_rhs(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    J_fn,
    quadrature_order: int = 3,
) -> np.ndarray:
    _validate_spaces_same_mesh(mesh, vector_space)

    ndofs = vector_space.ndofs
    f = np.zeros(ndofs, dtype=float)

    for cell_idx in range(mesh.n_cells):
        Fe = local_vector_source_rhs(
            mesh=mesh,
            cell_idx=cell_idx,
            J_fn=J_fn,
            quadrature_order=quadrature_order,
        )
        gdofs = vector_space.cell_dof_indices(cell_idx)
        sgn = vector_space.cell_dof_signs(cell_idx)

        for i in range(6):
            gi = gdofs[i]
            si = sgn[i]
            f[gi] += si * Fe[i]

    return f


def assemble_coulomb_coupling_matrix(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    quadrature_order: int = 2,
) -> np.ndarray:
    _validate_spaces_same_mesh(mesh, vector_space, scalar_space)

    nA = vector_space.ndofs
    nP = scalar_space.ndofs
    G = np.zeros((nA, nP), dtype=float)

    for cell_idx in range(mesh.n_cells):
        Ge = local_grad_p_coupling_matrix(
            mesh=mesh,
            cell_idx=cell_idx,
            quadrature_order=quadrature_order,
        )

        gdofs_A = vector_space.cell_dof_indices(cell_idx)
        sgn_A = vector_space.cell_dof_signs(cell_idx)
        gdofs_P = scalar_space.cell_dof_indices(cell_idx)

        for i in range(6):
            gi = gdofs_A[i]
            si = sgn_A[i]
            for j in range(4):
                gj = gdofs_P[j]
                G[gi, gj] += si * Ge[i, j]

    return G


def assemble_scalar_stiffness_matrix(
    mesh: TetraMesh,
    scalar_space: LagrangeP1Space,
    quadrature_order: int = 1,
) -> np.ndarray:
    if quadrature_order not in (1, 2, 3):
        raise ValueError("Supported quadrature orders are 1, 2 and 3.")

    if scalar_space.mesh is not mesh:
        raise ValueError("scalar_space must be built on the provided mesh.")

    nP = scalar_space.ndofs
    S = np.zeros((nP, nP), dtype=float)

    for cell_idx in range(mesh.n_cells):
        amap = AffineTetraMap(mesh.cell_vertices(cell_idx))
        grads = amap.physical_barycentric_gradients()
        q = get_tetra_quadrature(quadrature_order)
        detJ = amap.jacobian_determinant()

        Se = np.zeros((4, 4), dtype=float)
        for w in q.weights:
            for i in range(4):
                gi = grads[i]
                for j in range(4):
                    gj = grads[j]
                    Se[i, j] += float(np.dot(gi, gj)) * w * detJ

        gdofs = scalar_space.cell_dof_indices(cell_idx)
        for i in range(4):
            ii = gdofs[i]
            for j in range(4):
                jj = gdofs[j]
                S[ii, jj] += Se[i, j]

    return S


def assemble_discrete_gradient_matrix(
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
) -> np.ndarray:
    if vector_space.mesh is not scalar_space.mesh:
        raise ValueError("vector_space and scalar_space must be built on the same mesh.")

    nA = vector_space.ndofs
    nP = scalar_space.ndofs
    D = np.zeros((nA, nP), dtype=float)

    for e_idx, (i, j) in enumerate(vector_space.global_edges):
        D[e_idx, i] = -1.0
        D[e_idx, j] = +1.0

    return D


def assemble_mixed_coulomb_blocks(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    nu: float,
    J_fn,
    curl_quadrature_order: int = 1,
    coupling_quadrature_order: int = 2,
    rhs_quadrature_order: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_spaces_same_mesh(mesh, vector_space, scalar_space)

    K = np.zeros((vector_space.ndofs, vector_space.ndofs), dtype=float)
    G = np.zeros((vector_space.ndofs, scalar_space.ndofs), dtype=float)
    f = np.zeros(vector_space.ndofs, dtype=float)

    for cell_idx in range(mesh.n_cells):
        Ke = local_curlcurl_block(
            mesh=mesh,
            cell_idx=cell_idx,
            nu=nu,
            quadrature_order=curl_quadrature_order,
        )
        Ge = local_grad_p_coupling_matrix(
            mesh=mesh,
            cell_idx=cell_idx,
            quadrature_order=coupling_quadrature_order,
        )
        Fe = local_vector_source_rhs(
            mesh=mesh,
            cell_idx=cell_idx,
            J_fn=J_fn,
            quadrature_order=rhs_quadrature_order,
        )

        gdofs_A = vector_space.cell_dof_indices(cell_idx)
        sgn_A = vector_space.cell_dof_signs(cell_idx)
        gdofs_P = scalar_space.cell_dof_indices(cell_idx)

        for i in range(6):
            gi = gdofs_A[i]
            si = sgn_A[i]

            f[gi] += si * Fe[i]

            for j in range(6):
                gj = gdofs_A[j]
                sj = sgn_A[j]
                K[gi, gj] += si * sj * Ke[i, j]

            for j in range(4):
                gj = gdofs_P[j]
                G[gi, gj] += si * Ge[i, j]

    return K, G, f


def assemble_mixed_coulomb_system(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    nu: float,
    J_fn,
    curl_quadrature_order: int = 1,
    coupling_quadrature_order: int = 2,
    rhs_quadrature_order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    K, G, f = assemble_mixed_coulomb_blocks(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        nu=nu,
        J_fn=J_fn,
        curl_quadrature_order=curl_quadrature_order,
        coupling_quadrature_order=coupling_quadrature_order,
        rhs_quadrature_order=rhs_quadrature_order,
    )

    nA = vector_space.ndofs
    nP = scalar_space.ndofs

    A = np.zeros((nA + nP, nA + nP), dtype=float)
    b = np.zeros(nA + nP, dtype=float)

    A[:nA, :nA] = K
    A[:nA, nA:] = G
    A[nA:, :nA] = G.T

    b[:nA] = f

    return A, b