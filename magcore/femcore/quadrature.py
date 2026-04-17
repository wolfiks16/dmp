from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True, slots=True)
class TetraQuadratureRule:
    points: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float)
        w = np.asarray(self.weights, dtype=float)

        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "weights", w)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")
        if w.ndim != 1 or len(w) != len(pts):
            raise ValueError("weights must have shape (N,) and match points.")
        if not np.isfinite(pts).all() or not np.isfinite(w).all():
            raise ValueError("quadrature points and weights must be finite.")

        # Для правил более высокого порядка веса могут быть отрицательными.
        # Поэтому запрещаем только вырожденную сумму весов.
        if np.isclose(float(np.sum(w)), 0.0):
            raise ValueError("quadrature weights must have nonzero sum.")

        # Все точки должны лежать внутри эталонного тетраэдра:
        # x >= 0, y >= 0, z >= 0, x + y + z <= 1.
        if np.any(pts < 0.0):
            raise ValueError("quadrature points must lie inside the reference tetrahedron.")
        if np.any(np.sum(pts, axis=1) > 1.0 + 1e-14):
            raise ValueError("quadrature points must lie inside the reference tetrahedron.")

    @property
    def n_points(self) -> int:
        return len(self.weights)

    @property
    def weight_sum(self) -> float:
        return float(np.sum(self.weights))


def tetra_quadrature_1pt() -> TetraQuadratureRule:
    """
    1-точечное правило для эталонного тетраэдра.
    Точно для полиномов степени <= 1.
    """
    points = np.array([[0.25, 0.25, 0.25]], dtype=float)
    weights = np.array([1.0 / 6.0], dtype=float)
    return TetraQuadratureRule(points=points, weights=weights)


def tetra_quadrature_4pt() -> TetraQuadratureRule:
    """
    4-точечное правило Hammer для эталонного тетраэдра.
    Точно для полиномов степени <= 2.
    """
    sqrt5 = math.sqrt(5.0)
    a = (5.0 + 3.0 * sqrt5) / 20.0
    b = (5.0 - sqrt5) / 20.0

    points = np.array(
        [
            [b, b, b],
            [a, b, b],
            [b, a, b],
            [b, b, a],
        ],
        dtype=float,
    )
    weights = np.full(4, 1.0 / 24.0, dtype=float)
    return TetraQuadratureRule(points=points, weights=weights)


def tetra_quadrature_5pt() -> TetraQuadratureRule:
    """
    5-точечное правило Hammer–Stroud для эталонного тетраэдра.
    Точно для полиномов степени <= 3.

    В этом правиле центральный вес отрицателен — это нормально.
    """
    points = np.array(
        [
            [0.25, 0.25, 0.25],
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            [0.5, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 0.5, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 0.5],
        ],
        dtype=float,
    )

    weights = np.array(
        [
            -2.0 / 15.0,
            3.0 / 40.0,
            3.0 / 40.0,
            3.0 / 40.0,
            3.0 / 40.0,
        ],
        dtype=float,
    )

    return TetraQuadratureRule(points=points, weights=weights)


def get_tetra_quadrature(order: int = 2) -> TetraQuadratureRule:
    if order == 1:
        return tetra_quadrature_1pt()
    if order == 2:
        return tetra_quadrature_4pt()
    if order == 3:
        return tetra_quadrature_5pt()
    raise ValueError("Supported tetra quadrature orders are 1, 2 and 3.")