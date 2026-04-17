from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.femcore.assembly import (
    assemble_coulomb_coupling_matrix,
    assemble_discrete_gradient_matrix,
    assemble_mass_matrix,
    assemble_scalar_stiffness_matrix,
)
from magcore.femcore.boundary_conditions import apply_zero_dirichlet_bc
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh import TetraMesh


@dataclass(frozen=True, slots=True)
class GaugeProjectionResult:
    phi: np.ndarray
    a_parallel: np.ndarray
    a_perp: np.ndarray
    norm_total: float
    norm_parallel: float
    norm_perp: float
    eta_parallel: float


def gauge_residual_vector(coupling_matrix: np.ndarray, a: np.ndarray) -> np.ndarray:
    G = np.asarray(coupling_matrix, dtype=float)
    a = np.asarray(a, dtype=float)

    if G.ndim != 2:
        raise ValueError("coupling_matrix must be a 2D matrix.")
    if a.ndim != 1 or a.shape[0] != G.shape[0]:
        raise ValueError("a must have shape (coupling_matrix.shape[0],).")
    if not np.isfinite(G).all():
        raise ValueError("coupling_matrix must be finite.")
    if not np.isfinite(a).all():
        raise ValueError("a must be finite.")

    return G.T @ a


def gauge_residual_norm(coupling_matrix: np.ndarray, a: np.ndarray) -> float:
    r = gauge_residual_vector(coupling_matrix, a)
    return float(np.linalg.norm(r))


def _vector_l2_norm_from_mass(mass_matrix: np.ndarray, coeffs: np.ndarray) -> float:
    M = np.asarray(mass_matrix, dtype=float)
    x = np.asarray(coeffs, dtype=float)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("mass_matrix must be square.")
    if x.ndim != 1 or x.shape[0] != M.shape[0]:
        raise ValueError("coeffs must have shape (mass_matrix.shape[0],).")

    value = float(x @ (M @ x))
    value = max(value, 0.0)
    return float(np.sqrt(value))


def project_to_gradient_subspace(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    a: np.ndarray,
    coupling_quadrature_order: int = 2,
    scalar_stiffness_quadrature_order: int = 1,
    vector_mass_quadrature_order: int = 2,
) -> GaugeProjectionResult:
    if vector_space.mesh is not mesh or scalar_space.mesh is not mesh:
        raise ValueError("Both spaces must be built on the provided mesh.")

    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.shape[0] != vector_space.ndofs:
        raise ValueError("a must have shape (vector_space.ndofs,).")
    if not np.isfinite(a).all():
        raise ValueError("a must be finite.")

    G = assemble_coulomb_coupling_matrix(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        quadrature_order=coupling_quadrature_order,
    )
    S = assemble_scalar_stiffness_matrix(
        mesh=mesh,
        scalar_space=scalar_space,
        quadrature_order=scalar_stiffness_quadrature_order,
    )
    D = assemble_discrete_gradient_matrix(
        vector_space=vector_space,
        scalar_space=scalar_space,
    )

    rhs = G.T @ a
    scalar_bnd = scalar_space.boundary_dofs()
    S_bc, rhs_bc = apply_zero_dirichlet_bc(S, rhs, scalar_bnd)
    phi = np.linalg.solve(S_bc, rhs_bc)

    a_parallel = D @ phi
    a_perp = a - a_parallel

    M = assemble_mass_matrix(
        mesh=mesh,
        space=vector_space,
        alpha=1.0,
        quadrature_order=vector_mass_quadrature_order,
    )

    norm_total = _vector_l2_norm_from_mass(M, a)
    norm_parallel = _vector_l2_norm_from_mass(M, a_parallel)
    norm_perp = _vector_l2_norm_from_mass(M, a_perp)

    if norm_total > 0.0:
        eta_parallel = norm_parallel / norm_total
    else:
        eta_parallel = 0.0

    return GaugeProjectionResult(
        phi=phi,
        a_parallel=a_parallel,
        a_perp=a_perp,
        norm_total=norm_total,
        norm_parallel=norm_parallel,
        norm_perp=norm_perp,
        eta_parallel=eta_parallel,
    )