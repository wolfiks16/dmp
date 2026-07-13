from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ManufacturedCurrent:
    """Изготовленное решение (MMS) для линейной задачи −div(ν₀ ∇A_z) = J_z, ν₀=const."""

    A_exact: Callable[[np.ndarray], float]
    grad_A_exact: Callable[[np.ndarray], np.ndarray]
    J_fn: Callable[[np.ndarray], float]


def manufactured_sine_current(nu0: float = 1.0) -> ManufacturedCurrent:
    """
    A_z(x,y) = sin(πx)·sin(πy) на [0,1]² (обнуляется на границе ⇒ однородный Dirichlet).
    ΔA_z = −2π² A_z ⇒ J_z = −div(ν₀∇A_z) = 2π²ν₀·sin(πx)sin(πy). Оракул порядков: L²→2, H¹→1.
    """
    pi = math.pi

    def A_exact(x: np.ndarray) -> float:
        return math.sin(pi * x[0]) * math.sin(pi * x[1])

    def grad_A_exact(x: np.ndarray) -> np.ndarray:
        return np.array(
            [pi * math.cos(pi * x[0]) * math.sin(pi * x[1]),
             pi * math.sin(pi * x[0]) * math.cos(pi * x[1])],
            dtype=float,
        )

    def J_fn(x: np.ndarray) -> float:
        return 2.0 * pi * pi * nu0 * math.sin(pi * x[0]) * math.sin(pi * x[1])

    return ManufacturedCurrent(A_exact=A_exact, grad_A_exact=grad_A_exact, J_fn=J_fn)


@dataclass(frozen=True)
class ManufacturedNonlinear:
    """MMS для НЕЛИНЕЙНОЙ задачи −div(ν(|B|)∇A_z)=J_z с законом ν=ν₀(1+c|B|²)."""

    A_exact: Callable[[np.ndarray], float]
    grad_A_exact: Callable[[np.ndarray], np.ndarray]
    J_fn: Callable[[np.ndarray], float]
    nu_of_B: Callable[[np.ndarray], np.ndarray]   # (n_cells,2)->(n_cells,)


def manufactured_nonlinear_sine(nu0: float = 1.0, c: float = 0.5) -> ManufacturedNonlinear:
    """
    A_z=sin(πx)sin(πy) на [0,1]² (однородный Dirichlet), ν=ν₀(1+c·|B|²), |B|²=|∇A_z|².
    Аналитический источник (сатурационно-подобный закон, монотонный):
        J_z = −div(ν∇A_z) = −(∇ν·∇A_z + ν ΔA_z),
    с ΔA_z=−2π²A_z и ∇ν=ν₀c∇(|∇A_z|²). Оракул независимости решателя от закона (не тавтология).
    """
    pi = math.pi

    def A_exact(x: np.ndarray) -> float:
        return math.sin(pi * x[0]) * math.sin(pi * x[1])

    def grad_A_exact(x: np.ndarray) -> np.ndarray:
        return np.array(
            [pi * math.cos(pi * x[0]) * math.sin(pi * x[1]),
             pi * math.sin(pi * x[0]) * math.cos(pi * x[1])],
            dtype=float,
        )

    def J_fn(x: np.ndarray) -> float:
        sx, cx = math.sin(pi * x[0]), math.cos(pi * x[0])
        sy, cy = math.sin(pi * x[1]), math.cos(pi * x[1])
        A = sx * sy
        Ax = pi * cx * sy
        Ay = pi * sx * cy
        Axx = -pi * pi * sx * sy
        Ayy = -pi * pi * sx * sy
        Axy = pi * pi * cx * cy
        s = Ax * Ax + Ay * Ay
        lap = Axx + Ayy                                    # = −2π²A
        # ∇ν·∇A_z = ν₀c·(∂_x s·A_x + ∂_y s·A_y), ∂_x s=2(A_x A_xx+A_y A_xy), ∂_y s=2(A_x A_xy+A_y A_yy)
        grad_nu_dot_grad_A = nu0 * c * 2.0 * (
            Ax * Ax * Axx + 2.0 * Ax * Ay * Axy + Ay * Ay * Ayy
        )
        nu = nu0 * (1.0 + c * s)
        return -(grad_nu_dot_grad_A + nu * lap)

    def nu_of_B(B_cells: np.ndarray) -> np.ndarray:
        B = np.asarray(B_cells, dtype=float)
        s = B[:, 0] ** 2 + B[:, 1] ** 2
        return nu0 * (1.0 + c * s)

    return ManufacturedNonlinear(
        A_exact=A_exact, grad_A_exact=grad_A_exact, J_fn=J_fn, nu_of_B=nu_of_B
    )
