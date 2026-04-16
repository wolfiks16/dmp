from __future__ import annotations

import numpy as np

from magcore.femcore.spaces import NedelecP1Space


def find_boundary_dofs(space: NedelecP1Space) -> tuple[int, ...]:
    """
    For first-order edge elements with homogeneous tangential Dirichlet condition,
    boundary DOFs are the DOFs associated with boundary edges.
    """
    return space.boundary_dofs()


def apply_zero_dirichlet_bc(
    A: np.ndarray,
    b: np.ndarray,
    dofs: tuple[int, ...] | list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply homogeneous Dirichlet conditions by row/column elimination:
    - zero row and column
    - set diagonal to 1
    - set rhs entry to 0
    """
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