from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def apply_dirichlet(K, f, dofs, values=0.0):
    """
    Симметричное наложение Dirichlet на узлы `dofs` со значениями `values` (скаляр или
    массив длины len(dofs)). Возвращает (K_bc, f_bc); исходные не меняет. Полиморфно:
    плотная K (np.ndarray, масштаб верификации) и разрежённая K (scipy.sparse, реальные
    сетки) обрабатываются идентично по значениям.
    """
    dofs = np.asarray(list(dofs), dtype=int)
    f = np.array(f, dtype=float, copy=True)
    if dofs.size == 0:
        return K, f
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 0:
        vals = np.full(dofs.shape, float(vals))
    if vals.shape != dofs.shape:
        raise ValueError("values must be a scalar or match len(dofs).")

    if sp.issparse(K):
        n = K.shape[0]
        Kc = K.tocsr()
        # Перенести вклад закреплённых столбцов в правую часть (симметричная элиминация).
        f = f - np.asarray(Kc[:, dofs] @ vals).ravel()
        keep = np.ones(n, dtype=bool)
        keep[dofs] = False
        D = sp.diags(keep.astype(float))
        Kbc = (D @ Kc @ D).tocsr()                    # обнулить строки И столбцы dofs
        Kbc = Kbc + sp.coo_matrix(
            (np.ones(dofs.size), (dofs, dofs)), shape=(n, n)
        ).tocsr()                                     # единица на диагонали закреплённых
        Kbc.eliminate_zeros()
        f[dofs] = vals
        return Kbc, f

    # Плотный путь (масштаб верификации) — без изменений.
    K = np.array(K, dtype=float, copy=True)
    f = f - K[:, dofs] @ vals
    for d, v in zip(dofs, vals):
        K[d, :] = 0.0
        K[:, d] = 0.0
        K[d, d] = 1.0
        f[d] = v
    return K, f


def solve_scalar(K, f):
    """Прямое решение SPD-системы P1: разрежённая → scipy spsolve; плотная → numpy."""
    if sp.issparse(K):
        return spsolve(K.tocsr(), np.asarray(f, dtype=float))
    return np.linalg.solve(K, f)
