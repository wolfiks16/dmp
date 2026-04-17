from __future__ import annotations

import numpy as np

from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space


def find_boundary_dofs(space: NedelecP1Space) -> tuple[int, ...]:
    return space.boundary_dofs()


def apply_zero_dirichlet_bc(
    A: np.ndarray,
    b: np.ndarray,
    dofs: tuple[int, ...] | list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    dofs = np.asarray(dofs, dtype=int)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must have shape (A.shape[0],).")
    n = A.shape[0]
    if np.any(dofs < 0) or np.any(dofs >= n):
        raise ValueError("Boundary dofs contain out-of-range indices.")
    for d in dofs:
        A[d, :] = 0.0
        A[:, d] = 0.0
        A[d, d] = 1.0
        b[d] = 0.0
    return A, b


def find_mixed_boundary_dofs(
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return vector_space.boundary_dofs(), scalar_space.boundary_dofs()


def apply_zero_dirichlet_bc_mixed(
    A: np.ndarray,
    b: np.ndarray,
    vector_dofs: tuple[int, ...] | list[int] | np.ndarray,
    scalar_dofs: tuple[int, ...] | list[int] | np.ndarray,
    n_vector_dofs: int,
) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must have shape (A.shape[0],).")
    if not (0 < n_vector_dofs < A.shape[0]):
        raise ValueError("n_vector_dofs must split the mixed system into vector/scalar blocks.")

    vd = np.asarray(vector_dofs, dtype=int)
    sd = np.asarray(scalar_dofs, dtype=int) + int(n_vector_dofs)
    all_dofs = np.concatenate((vd, sd))
    if np.any(all_dofs < 0) or np.any(all_dofs >= A.shape[0]):
        raise ValueError("Boundary dofs contain out-of-range indices.")
    for d in all_dofs:
        A[d, :] = 0.0
        A[:, d] = 0.0
        A[d, d] = 1.0
        b[d] = 0.0
    return A, b
