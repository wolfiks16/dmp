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


def manufactured_nonlinear_chord_nu(B_cells: np.ndarray, nu0: float, c: float) -> np.ndarray:
    """
    Хордовая релуктивность нелинейного MMS: ν(|B|) = ν0·(1 + c·|B|²), по ячейке.
    Согласована с источником `manufactured_nonlinear_rhs` (тот же закон).
    """
    if nu0 <= 0.0:
        raise ValueError("nu0 must be positive.")
    if c < 0.0:
        raise ValueError("c must be non-negative.")
    B = np.asarray(B_cells, dtype=float)
    if B.ndim != 2 or B.shape[1] != 3:
        raise ValueError("B_cells must have shape (n_cells, 3).")
    return nu0 * (1.0 + c * np.sum(B * B, axis=1))


def manufactured_nonlinear_rhs(x: np.ndarray, nu0: float, c: float) -> np.ndarray:
    """
    RHS для НЕЛИНЕЙНОГО manufactured-решения с законом ν(|B|)=ν0(1+c·|B|²):

        curl( ν(|curl A_ref|) · curl A_ref ) = J,   B = curl A_ref = (0, f_z, −f_y).

    Так как A_ref=(f(y,z),0,0) и H=νB не зависят от x, остаётся только x-компонента:
        J = (J_x, 0, 0),  J_x = g·C − 2·g'·D,
    где g=ν0(1+cS), g'=ν0·c, S=f_y²+f_z²,
        C = 2z(1−z)+2y(1−y)  (= −(f_yy+f_zz), множитель линейного curl-curl),
        D = f_y²·f_yy + f_z²·f_zz + 2·f_y·f_z·f_yz  (вклад ∇ν×B).
    Вывод аналитический (см. docs/math/nonlinear_materials.md §5). При c=0 сводится к
    линейному `ν0·manufactured_curlcurl_ref`.
    """
    if nu0 <= 0.0:
        raise ValueError("nu0 must be positive.")
    if c < 0.0:
        raise ValueError("c must be non-negative.")
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")

    y = float(x[1])
    z = float(x[2])
    f_y = (1.0 - 2.0 * y) * z * (1.0 - z)
    f_z = y * (1.0 - y) * (1.0 - 2.0 * z)
    f_yy = -2.0 * z * (1.0 - z)
    f_zz = -2.0 * y * (1.0 - y)
    f_yz = (1.0 - 2.0 * y) * (1.0 - 2.0 * z)

    C = 2.0 * z * (1.0 - z) + 2.0 * y * (1.0 - y)  # = −(f_yy+f_zz)
    S = f_y * f_y + f_z * f_z
    g = nu0 * (1.0 + c * S)
    gp = nu0 * c
    D = f_y * f_y * f_yy + f_z * f_z * f_zz + 2.0 * f_y * f_z * f_yz

    j_x = g * C - 2.0 * gp * D
    return np.array([j_x, 0.0, 0.0], dtype=float)


# --- x-ПЕРИОДИЧЕСКОЕ manufactured-решение (для верификации периодических ГУ) ---
# A = (0, g(x)·z(1−z), 0), g(x)=cos(2πx): x-периодично (период 1), tangential A=0 на
# гранях y,z (A_x=A_z=0; A_y=0 при z=0,1), div A=0 ⇒ p=0. Ненулевая тангенциальная A_y
# на гранях x=0,1 (=z(1−z)) ⇒ реально нагружает периодическую связь.

_TWO_PI = 2.0 * np.pi


def manufactured_periodic_A(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")
    xx, _, z = float(x[0]), float(x[1]), float(x[2])
    return np.array([0.0, np.cos(_TWO_PI * xx) * z * (1.0 - z), 0.0], dtype=float)


def manufactured_periodic_curl(x: np.ndarray) -> np.ndarray:
    """B = curl A = (−g(x)(1−2z), 0, g'(x)·z(1−z)), g'=−2π sin(2πx)."""
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")
    xx, _, z = float(x[0]), float(x[1]), float(x[2])
    g = np.cos(_TWO_PI * xx)
    gp = -_TWO_PI * np.sin(_TWO_PI * xx)
    return np.array([-g * (1.0 - 2.0 * z), 0.0, gp * z * (1.0 - z)], dtype=float)


def manufactured_periodic_rhs(x: np.ndarray, nu: float) -> np.ndarray:
    """
    J = curl(ν curl A) = ν·curl B = (0, ν·(2g − g''·z(1−z)), 0), g''=−4π²cos(2πx)
      ⇒ J_y = ν·cos(2πx)·(2 + 4π²·z(1−z)).  (const ν.)
    """
    if nu <= 0.0:
        raise ValueError("nu must be positive.")
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("x must have shape (3,).")
    xx, _, z = float(x[0]), float(x[1]), float(x[2])
    g = np.cos(_TWO_PI * xx)
    j_y = nu * g * (2.0 + 4.0 * np.pi * np.pi * z * (1.0 - z))
    return np.array([0.0, j_y, 0.0], dtype=float)