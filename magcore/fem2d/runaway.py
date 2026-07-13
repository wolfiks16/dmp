from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from magcore.fem2d.assembly import assemble_mass, assemble_stiffness
from magcore.fem2d.solver import solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import assemble_robin_boundary, assemble_source_rhs

# Термическая неустойчивость (thermal runaway) — ядро К6′ в линеаризованной форме.
# Потери с температурной обратной связью q(T) = q0 + s·(T − T_amb) (s = чувствительность:
# рост меди по ρ(T) и/или размагничивание, требующее компенсации). Связанная стационарная
# тепловая задача: −div(k∇T) − s·T = q0 − s·T_amb + конвекция ⇒ (K + R − s·M)·T = f.
# Порог runaway s_crit = наименьшее ОБОБЩЁННОЕ собственное значение (K+R)v = s·M·v:
#   s < s_crit ⇒ оператор SPD (устойчивый баланс, конечная T);
#   s > s_crit ⇒ определённость теряется (охлаждение не догоняет потери) ⇒ разгон.
# Тот же порог управляет сходимостью Picard T_{n+1}=(K+R)⁻¹(f + s·M·T_n).


@dataclass(frozen=True, slots=True)
class RunawayResult:
    T: np.ndarray                 # (ndofs,) поле температуры (последняя итерация)
    converged: bool               # False ⇒ разгон (runaway)
    n_iterations: int
    norm_history: tuple[float, ...]


def thermal_operator(space: LagrangeP1Space2D, k, h: float) -> np.ndarray:
    """Тепловой оператор с конвекцией: A = K(k) + R_robin(h) (SPD при h>0)."""
    K = assemble_stiffness(space, k)
    R, _ = assemble_robin_boundary(space, float(h))
    return K + R


def runaway_threshold(space: LagrangeP1Space2D, k, h: float) -> float:
    """
    Критическая чувствительность потерь s_crit = наименьшее обобщённое собств. значение
    (K+R)v = s·M·v (M — матрица масс). При s ≥ s_crit система разгоняется.
    """
    A = thermal_operator(space, k, h)
    M = assemble_mass(space)
    w = eigh(A, M, eigvals_only=True)   # A,M симметричны, M SPD
    return float(np.min(w))


def solve_thermal_steady_with_feedback(
    space: LagrangeP1Space2D,
    k,
    *,
    q0_cells: np.ndarray,
    loss_sensitivity: float,
    h: float,
    T_amb: float,
) -> np.ndarray:
    """
    Прямое решение связанной стационарной задачи (K+R−s·M)·T = f для s < s_crit.
    f = источник (q0 − s·T_amb по ячейке) + конвективная нагрузка T_amb.
    """
    s = float(loss_sensitivity)
    A = thermal_operator(space, k, h)
    M = assemble_mass(space)
    _, amb_load = assemble_robin_boundary(space, float(h))
    q0 = np.asarray(q0_cells, dtype=float)
    f = assemble_source_rhs(space, q0 - s * T_amb) + float(T_amb) * amb_load
    return solve_scalar(A - s * M, f)


def couple_thermal_loss_picard(
    space: LagrangeP1Space2D,
    k,
    *,
    q0_cells: np.ndarray,
    loss_sensitivity: float,
    h: float,
    T_amb: float,
    max_iter: int = 200,
    tol: float = 1.0e-8,
    divergence_cap: float = 1.0e6,
) -> RunawayResult:
    """
    Итерация связки потери↔температура: T_{n+1} = (K+R)⁻¹(f(q0 + s(T_n − T_amb)) + Robin).
    Сходится ⟺ s < s_crit (спектр. радиус s/s_crit < 1); иначе ‖T‖ растёт → runaway.
    Детект разгона: ‖T‖ превысила cap · масштаб T_amb.
    """
    s = float(loss_sensitivity)
    A = thermal_operator(space, k, h)
    M = assemble_mass(space)
    _, amb_load = assemble_robin_boundary(space, float(h))
    q0 = np.asarray(q0_cells, dtype=float)
    scale = max(abs(float(T_amb)), 1.0)

    # Постоянная часть f0 = ∫(q0 − s·T_amb)v + конвекция; член обратной связи s·M·T
    # (консистентная масса) ⇒ неподвижная точка A·T = f0 + s·M·T = прямое (A−sM)T=f0.
    f0 = assemble_source_rhs(space, q0 - s * T_amb) + float(T_amb) * amb_load

    T = np.full(space.ndofs, float(T_amb), dtype=float)
    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        T_new = solve_scalar(A, f0 + s * (M @ T))

        nrm = float(np.linalg.norm(T_new))
        history.append(nrm)
        if nrm > divergence_cap * scale * np.sqrt(space.ndofs):
            T = T_new
            converged = False
            break
        rel = float(np.linalg.norm(T_new - T)) / max(np.linalg.norm(T), 1e-30)
        T = T_new
        if rel < tol:
            converged = True
            break

    return RunawayResult(
        T=T, converged=converged, n_iterations=it, norm_history=tuple(history)
    )
