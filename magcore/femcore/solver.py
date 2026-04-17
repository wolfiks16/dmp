from __future__ import annotations

import numpy as np


def _validate_dense_linear_system(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def solve_dense_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить плотную линейную систему прямым методом.
    """
    A, b = _validate_dense_linear_system(A, b)
    return np.linalg.solve(A, b)


def solve_linear_hcurl_problem(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Решить стандартную H(curl)-FEM систему.
    """
    return solve_dense_linear_system(A, b)


def solve_mixed_coulomb_problem(system_matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Решить mixed saddle-point систему Кулона:
        [K  G ][a] = [f]
        [G^T 0][p]   [0]
    """
    return solve_dense_linear_system(system_matrix, rhs)


def split_mixed_solution(x: np.ndarray, n_vector_dofs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Разделить решение mixed-системы на:
    - a: коэффициенты A_h
    - p: коэффициенты p_h
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1D vector.")
    if not (0 <= n_vector_dofs <= x.shape[0]):
        raise ValueError("n_vector_dofs must be in [0, len(x)].")

    a = x[:n_vector_dofs].copy()
    p = x[n_vector_dofs:].copy()
    return a, p