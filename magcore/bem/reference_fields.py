from __future__ import annotations

import numpy as np


def harmonic_linear_x(x: np.ndarray) -> float:
    return float(x[0])


def harmonic_linear_y(x: np.ndarray) -> float:
    return float(x[1])


def harmonic_linear_z(x: np.ndarray) -> float:
    return float(x[2])


def harmonic_xy(x: np.ndarray) -> float:
    return float(x[0] * x[1])


def harmonic_x2_minus_y2(x: np.ndarray) -> float:
    return float(x[0] ** 2 - x[1] ** 2)


def harmonic_point_source(center: np.ndarray):
    """
    Return a decaying harmonic reference field in the exterior region:
        u(x) = 1 / (4*pi*|x-center|)

    This is harmonic away from the source point. When the source point is placed
    strictly inside the closed surface, the function is harmonic in the exterior
    domain and decays like O(1/r), which is appropriate for a pure single-layer
    exterior verification solve.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,).")

    def _ref(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (3,):
            raise ValueError("x must have shape (3,).")

        r = np.linalg.norm(x - center)
        if r == 0.0:
            raise ValueError("Point-source reference is singular at the source point.")
        return float(1.0 / (4.0 * np.pi * r))

    return _ref


def evaluate_reference_on_points(points: np.ndarray, ref_fn) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    return np.array([ref_fn(p) for p in points], dtype=float)