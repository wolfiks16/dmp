from __future__ import annotations

import numpy as np


def apply_dirichlet(
    K: np.ndarray, f: np.ndarray, dofs, values=0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Симметричное наложение Dirichlet на узлы `dofs` со значениями `values`
    (скаляр или массив длины len(dofs)). Возвращает (K_bc, f_bc); исходные не меняет.
    """
    K = np.array(K, dtype=float, copy=True)
    f = np.array(f, dtype=float, copy=True)
    dofs = np.asarray(list(dofs), dtype=int)
    if dofs.size == 0:
        return K, f
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 0:
        vals = np.full(dofs.shape, float(vals))
    if vals.shape != dofs.shape:
        raise ValueError("values must be a scalar or match len(dofs).")

    # Перенести вклад закреплённых узлов в правую часть (симметричная элиминация).
    f = f - K[:, dofs] @ vals
    for d, v in zip(dofs, vals):
        K[d, :] = 0.0
        K[:, d] = 0.0
        K[d, d] = 1.0
        f[d] = v
    return K, f


def solve_scalar(K: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Прямое решение SPD-системы P1 (масштаб верификации)."""
    return np.linalg.solve(K, f)
