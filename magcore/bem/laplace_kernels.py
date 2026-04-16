from __future__ import annotations

import math
import numpy as np


FOUR_PI = 4.0 * math.pi


def laplace_green_3d(x: np.ndarray, y: np.ndarray) -> float:
    """
    3D Laplace fundamental solution:
        G(x, y) = 1 / (4*pi*|x-y|)
    """
    r = np.linalg.norm(x - y)
    if r == 0.0:
        raise ValueError("Laplace kernel is singular for coincident points.")
    return 1.0 / (FOUR_PI * float(r))


def laplace_green_3d_safe(x: np.ndarray, y: np.ndarray, eps: float = 0.0) -> float:
    """
    Safe variant with explicit lower-distance guard.
    """
    r = np.linalg.norm(x - y)
    if r <= eps:
        raise ValueError("Laplace kernel encountered near-singular or singular distance.")
    return 1.0 / (FOUR_PI * float(r))