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
    """
    Degree-1 exact quadrature on the reference tetrahedron.

    Reference tetra volume = 1/6.
    """
    points = np.array([[0.25, 0.25, 0.25]], dtype=float)
    weights = np.array([1.0 / 6.0], dtype=float)
    return TetraQuadratureRule(points=points, weights=weights)


def tetra_quadrature_4pt() -> TetraQuadratureRule:
    """
    Symmetric 4-point quadrature on the reference tetrahedron.

    Degree-2 exact rule with barycentric permutations of (a, b, b, b),
    where:
        a = (5 + 3*sqrt(5)) / 20
        b = (5 - sqrt(5)) / 20

    In reference coordinates (x, y, z), these become permutations where
    x = lambda_1, y = lambda_2, z = lambda_3.

    Each weight is 1/24, so the weights sum to 1/6.
    """
    sqrt5 = math.sqrt(5.0)
    a = (5.0 + 3.0 * sqrt5) / 20.0
    b = (5.0 - sqrt5) / 20.0

    # Barycentric permutations:
    # (a,b,b,b), (b,a,b,b), (b,b,a,b), (b,b,b,a)
    # Reference coordinates are (lambda1, lambda2, lambda3).
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
    """
    Return a tetrahedron quadrature rule.

    Supported orders:
        1 -> 1-point rule
        2 -> 4-point rule
    """
    if order == 1:
        return tetra_quadrature_1pt()
    if order == 2:
        return tetra_quadrature_4pt()
    raise ValueError("Supported tetra quadrature orders are 1 and 2.")