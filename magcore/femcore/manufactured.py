from __future__ import annotations

import numpy as np


def manufactured_A_ref(x: np.ndarray) -> np.ndarray:
    """
    Manufactured vector potential on the unit cube:

        A(x,y,z) = [ y(1-y) z(1-z), 0, 0 ]

    This satisfies homogeneous tangential Dirichlet condition
    n x A = 0 on the boundary of the unit cube:
    - on y=0,1 or z=0,1 => A = 0
    - on x=0,1 => A is parallel to the outward normal
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")

    y = float(x[1])
    z = float(x[2])
    f = y * (1.0 - y) * z * (1.0 - z)
    return np.array([f, 0.0, 0.0], dtype=float)


def manufactured_curl_ref(x: np.ndarray) -> np.ndarray:
    """
    Curl of manufactured_A_ref:

        A = [f(y,z), 0, 0]
        curl A = [0, d f/dz, -d f/dy]
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")

    y = float(x[1])
    z = float(x[2])

    df_dz = y * (1.0 - y) * (1.0 - 2.0 * z)
    df_dy = (1.0 - 2.0 * y) * z * (1.0 - z)

    return np.array([0.0, df_dz, -df_dy], dtype=float)


def manufactured_curlcurl_ref(x: np.ndarray) -> np.ndarray:
    """
    Curl-curl of manufactured_A_ref.

    For:
        A = [f(y,z), 0, 0]
    we have:
        curl curl A = [ -d^2f/dy^2 - d^2f/dz^2, 0, 0 ]

    where:
        d^2f/dy^2 = -2 z(1-z)
        d^2f/dz^2 = -2 y(1-y)

    hence:
        curl curl A = [ 2 z(1-z) + 2 y(1-y), 0, 0 ].
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")

    y = float(x[1])
    z = float(x[2])

    val = 2.0 * z * (1.0 - z) + 2.0 * y * (1.0 - y)
    return np.array([val, 0.0, 0.0], dtype=float)


def manufactured_rhs(x: np.ndarray, nu: float, alpha: float) -> np.ndarray:
    """
    Right-hand side for the manufactured H(curl) problem:

        curl(nu curl A_ref) + alpha A_ref = f

    with constant nu and alpha.
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    return nu * manufactured_curlcurl_ref(x) + alpha * manufactured_A_ref(x)