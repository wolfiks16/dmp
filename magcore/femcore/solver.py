from __future__ import annotations

import numpy as np


def _validate_linear_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    return A, b


def solve_linear_hcurl_problem(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A, b = _validate_linear_system(A, b)
    return np.linalg.solve(A, b)


def solve_mixed_linear_problem(A: np.ndarray, b: np.ndarray, n_vector_dofs: int) -> tuple[np.ndarray, np.ndarray]:
    A, b = _validate_linear_system(A, b)
    if not (0 < n_vector_dofs < A.shape[0]):
        raise ValueError("n_vector_dofs must split the mixed system into vector/scalar blocks.")
    x = np.linalg.solve(A, b)
    return x[:n_vector_dofs], x[n_vector_dofs:]
