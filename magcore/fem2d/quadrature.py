from __future__ import annotations

import numpy as np

# Симметричные квадратуры на опорном треугольнике в барицентрических координатах
# (l0,l1,l2), веса нормированы Σw=1 (физический интеграл = Area·Σ w_q f(x_q)).


def _deg1() -> tuple[np.ndarray, np.ndarray]:
    bary = np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=float)
    w = np.array([1.0], dtype=float)
    return bary, w


def _deg2() -> tuple[np.ndarray, np.ndarray]:
    # 3-точечная (точна для полиномов степени 2).
    a, b = 2 / 3, 1 / 6
    bary = np.array([[a, b, b], [b, a, b], [b, b, a]], dtype=float)
    w = np.full(3, 1 / 3, dtype=float)
    return bary, w


def _deg5() -> tuple[np.ndarray, np.ndarray]:
    # 7-точечная (Dunavant, степень 5, все веса положительны).
    a1 = 0.470142064105115
    a2 = 0.101286507323456
    w0 = 0.225
    w1 = 0.132394152788506
    w2 = 0.125939180544827
    bary = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1 - 2 * a1, a1, a1], [a1, 1 - 2 * a1, a1], [a1, a1, 1 - 2 * a1],
            [1 - 2 * a2, a2, a2], [a2, 1 - 2 * a2, a2], [a2, a2, 1 - 2 * a2],
        ],
        dtype=float,
    )
    w = np.array([w0, w1, w1, w1, w2, w2, w2], dtype=float)
    return bary, w


def triangle_quadrature(order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Барицентрические точки (Q,3) и веса (Q,) (Σw=1) для интегрирования по треугольнику.
    order<=1 → степень 1; order==2 → степень 2; иначе → степень 5 (безопасно для MMS).
    """
    if order <= 1:
        return _deg1()
    if order == 2:
        return _deg2()
    return _deg5()
