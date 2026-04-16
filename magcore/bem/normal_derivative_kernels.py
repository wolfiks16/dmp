from __future__ import annotations

import math
import numpy as np


FOUR_PI = 4.0 * math.pi


def laplace_dgreen_dn_x(x: np.ndarray, y: np.ndarray, n_x: np.ndarray) -> float:
    """
    Normal derivative of the 3D Laplace Green's function with respect to the
    target point x along normal n_x:

        G(x, y) = 1 / (4*pi*|x-y|)

        dG/dn_x = grad_x G(x, y) · n_x
                = - ((x-y) · n_x) / (4*pi*|x-y|^3)

    Parameters
    ----------
    x:
        Target point, shape (3,)
    y:
        Source point, shape (3,)
    n_x:
        Unit normal at the target point, shape (3,)

    Returns
    -------
    float
        Normal derivative with respect to x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x = np.asarray(n_x, dtype=float)

    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")
    if y.shape != (3,):
        raise ValueError("y must have shape (3,).")
    if n_x.shape != (3,):
        raise ValueError("n_x must have shape (3,).")

    r_vec = x - y
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("Normal derivative kernel is singular for coincident x and y.")

    return -float(np.dot(r_vec, n_x)) / (FOUR_PI * r**3)


def laplace_dgreen_dn_y(x: np.ndarray, y: np.ndarray, n_y: np.ndarray) -> float:
    """
    Normal derivative of the 3D Laplace Green's function with respect to the
    source point y along normal n_y:

        dG/dn_y = grad_y G(x, y) · n_y
                = + ((x-y) · n_y) / (4*pi*|x-y|^3)

    because grad_y G = -grad_x G.

    Parameters
    ----------
    x:
        Target point, shape (3,)
    y:
        Source point, shape (3,)
    n_y:
        Unit normal at the source point, shape (3,)

    Returns
    -------
    float
        Normal derivative with respect to y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_y = np.asarray(n_y, dtype=float)

    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")
    if y.shape != (3,):
        raise ValueError("y must have shape (3,).")
    if n_y.shape != (3,):
        raise ValueError("n_y must have shape (3,).")

    r_vec = x - y
    r = float(np.linalg.norm(r_vec))
    if r == 0.0:
        raise ValueError("Normal derivative kernel is singular for coincident x and y.")

    return float(np.dot(r_vec, n_y)) / (FOUR_PI * r**3)