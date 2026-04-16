from __future__ import annotations

import numpy as np


def linear_background_potential(H0: np.ndarray):
    """
    Return scalar potential of a uniform background magnetic field H0.

    Convention:
        H = -grad(phi)
    therefore for constant H0:
        phi_bg(x) = -H0 · x
    """
    H0 = np.asarray(H0, dtype=float)
    if H0.shape != (3,):
        raise ValueError("H0 must have shape (3,).")

    def _phi(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (3,):
            raise ValueError("x must have shape (3,).")
        return float(-np.dot(H0, x))

    return _phi


def linear_background_gradient(H0: np.ndarray) -> np.ndarray:
    """
    Gradient of the scalar potential phi_bg(x) = -H0 · x.

    Since:
        grad(phi_bg) = -H0
    """
    H0 = np.asarray(H0, dtype=float)
    if H0.shape != (3,):
        raise ValueError("H0 must have shape (3,).")
    return -H0.copy()


def linear_background_normal_flux(H0: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Compute normal derivative of the background potential on a set of normals:

        q_bg = d(phi_bg)/dn = grad(phi_bg) · n = (-H0) · n

    Parameters
    ----------
    H0:
        Background magnetic field vector, shape (3,)
    normals:
        Array of unit normals, shape (N, 3)

    Returns
    -------
    q_bg:
        Array of shape (N,)
    """
    H0 = np.asarray(H0, dtype=float)
    normals = np.asarray(normals, dtype=float)

    if H0.shape != (3,):
        raise ValueError("H0 must have shape (3,).")
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (N, 3).")

    grad_phi = linear_background_gradient(H0)
    return normals @ grad_phi


def evaluate_background_on_points(points: np.ndarray, bg_fn) -> np.ndarray:
    """
    Evaluate a scalar background potential function on points of shape (N, 3).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")

    return np.array([bg_fn(p) for p in points], dtype=float)