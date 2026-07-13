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
