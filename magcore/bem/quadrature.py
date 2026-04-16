from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


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


def select_quadrature_strategy(source_face: int, target_face: int) -> QuadratureStrategy:
    """
    Minimal placeholder policy for now.
    """
    if source_face == target_face:
        return QuadratureStrategy.SINGULAR
    return QuadratureStrategy.REGULAR