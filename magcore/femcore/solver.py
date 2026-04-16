from __future__ import annotations

import numpy as np


def solve_linear_hcurl_problem(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve a linear H(curl)-FEM system with a dense direct solver.

    Parameters
    ----------
    A:
        Square system matrix, shape (N, N)
    b:
        Right-hand side vector, shape (N,)

    Returns
    -------
    x:
        Solution vector, shape (N,)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must have shape (A.shape[0],).")
    if not np.isfinite(A).all():
        raise ValueError("A must be finite.")
    if not np.isfinite(b).all():
        raise ValueError("b must be finite.")

    return np.linalg.solve(A, b)