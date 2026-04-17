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
        if np.any(w <= 0.0):
            raise ValueError("quadrature weights must be positive.")

    @property
    def n_points(self) -> int:
        return len(self.weights)

    @property
    def weight_sum(self) -> float:
        return float(np.sum(self.weights))


def tetra_quadrature_1pt() -> TetraQuadratureRule:
    points = np.array([[0.25, 0.25, 0.25]], dtype=float)
    weights = np.array([1.0 / 6.0], dtype=float)
    return TetraQuadratureRule(points=points, weights=weights)


def tetra_quadrature_4pt() -> TetraQuadratureRule:
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


def get_tetra_quadrature(order: int = 2) -> TetraQuadratureRule:
    if order == 1:
        return tetra_quadrature_1pt()
    if order == 2:
        return tetra_quadrature_4pt()
    raise ValueError("Supported tetra quadrature orders are 1 and 2.")
