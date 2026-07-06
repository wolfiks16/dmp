from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.polynomial.legendre import leggauss


class QuadratureStrategy(str, Enum):
    REGULAR = "regular"
    NEAR_SINGULAR = "near_singular"
    SINGULAR = "singular"


@dataclass(frozen=True, slots=True)
class QuadratureRule:
    points: np.ndarray   # shape (N, 2) on reference triangle
    weights: np.ndarray  # shape (N,)
    order: int


def get_triangle_quadrature(order: int) -> QuadratureRule:
    """
    Quadrature rules on the reference triangle:
        T_hat = {(xi, eta): xi >= 0, eta >= 0, xi + eta <= 1}

    Weights integrate over the reference triangle whose area is 1/2.
    """
    if order <= 0:
        raise ValueError("Quadrature order must be positive.")

    if order == 1:
        # 1-point centroid rule
        points = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        weights = np.array([0.5], dtype=float)
        return QuadratureRule(points=points, weights=weights, order=1)

    # order >= 2 -> use simple 3-point rule
    points = np.array(
        [
            [1.0 / 6.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=float,
    )
    weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float)
    return QuadratureRule(points=points, weights=weights, order=2)


_COLLAPSED_CACHE: dict[int, QuadratureRule] = {}


def triangle_collapsed_gauss(order: int) -> QuadratureRule:
    """
    Тензорное правило Гаусса–Лежандра order×order на опорном треугольнике
    T_hat = {(ξ,η): ξ,η≥0, ξ+η≤1} через коллапс-отображение (Дюффи) единичного
    квадрата: ξ=a, η=(1−a)b, якобиан (1−a). Веса суммируются к 1/2 (площадь T_hat).

    Для ГЛАДКИХ (несингулярных) интегрантов — дальние/near непланарные пары BEM,
    где ядро ограничено: высокий порядок даёт точность без подразбиения.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if order in _COLLAPSED_CACHE:
        return _COLLAPSED_CACHE[order]
    x, w = leggauss(int(order))
    a = 0.5 * (x + 1.0)
    wa = 0.5 * w
    A, B = np.meshgrid(a, a, indexing="ij")
    WA, WB = np.meshgrid(wa, wa, indexing="ij")
    xi = A
    eta = (1.0 - A) * B
    jac = 1.0 - A
    points = np.column_stack([xi.ravel(), eta.ravel()])
    weights = (WA * WB * jac).ravel()
    rule = QuadratureRule(points=points, weights=weights, order=2 * int(order) - 1)
    _COLLAPSED_CACHE[order] = rule
    return rule


def select_quadrature_strategy(source_face: int, target_face: int) -> QuadratureStrategy:
    """
    Minimal placeholder policy for now.
    """
    if source_face == target_face:
        return QuadratureStrategy.SINGULAR
    return QuadratureStrategy.REGULAR