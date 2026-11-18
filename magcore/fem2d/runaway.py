from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, splu

from magcore.fem2d.assembly import assemble_mass_sparse, assemble_stiffness_sparse
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
#
# В РЕАЛЬНОЙ МАШИНЕ обратная связь ЛОКАЛИЗОВАНА: s(x) = ρ₀·α·|J(x)|² только в меди (`losses.copper_loss_sensitivity`),
# в стали, магните и воздухе её нет. Тогда порог — не число s, а множитель запаса λ_min обобщённой задачи
#   (K + R)·v = λ·M_s·v,   M_s,ij = ∫ s(x)·φ_i·φ_j dx,
# устойчиво ⟺ λ_min > 1 (`runaway_margin`). R — те же граничные условия Робена, что в транзиенте (для машины —
# дифференцированные: расточка → рама h_in, ротор → среда h_out; `thermal.assemble_robin_multi`). Однородное s на
# всю область — частный случай: λ_min = s_crit/s (`runaway_threshold`, верификационная фикстура). Сравнивать s меди
# с s_crit всей области нельзя: масса там по всей площади, а источник — только в пазах (порог занижен в разы).
#
# Связь с транзиентом (`coupled_transient`): шаг неявного Эйлера с ЯВНЫМ источником q^n = q(T^n),
#   (C/dt + K + R)·T^{n+1} = (C/dt + S)·T^n + …,
# имеет множитель моды μ из (C/dt + S)v = μ·(C/dt + K + R)v, и μ < 1 ⟺ vᵀSv < vᵀ(K+R)v — то есть ровно λ_min > 1,
# при ЛЮБОМ dt (дискретный порог совпадает с порогом стационара). S — тот же источник, собранный так, как его
# применяет шаг (q по средней температуре ячейки, ∫q·φ_i = q·A/3); от M_s он отличается на O(h²).


@dataclass(frozen=True, slots=True)
class RunawayResult:
    T: np.ndarray                 # (ndofs,) поле температуры (последняя итерация)
    converged: bool               # False ⇒ разгон (runaway)
    n_iterations: int
    norm_history: tuple[float, ...]


def _robin_matrix(space: LagrangeP1Space2D, h, robin) -> sp.csr_matrix:
    """Краевая матрица Робена: готовая `robin` ((R, rhs) как у `assemble_robin_multi` или сама R) либо однородное h."""
    if robin is not None:
        if h is not None:
            raise ValueError("граничные условия — либо h (однородно), либо robin (готовая краевая матрица).")
        R = robin[0] if isinstance(robin, tuple) else robin
        R = sp.csr_matrix(R, dtype=float)
        if R.shape != (space.ndofs, space.ndofs):
            raise ValueError("robin: краевая матрица должна быть (ndofs, ndofs).")
        return R
    if h is None:
        raise ValueError("нужны граничные условия: h или robin.")
    R, _ = assemble_robin_boundary(space, float(h))
    return R


def thermal_operator(space: LagrangeP1Space2D, k, h: float | None = None, *, robin=None) -> sp.csr_matrix:
    """
    Тепловой оператор с конвекцией: A = K(k) + R (SPD при h>0), разрежённый (CSR). R — однородная конвекция `h`
    на всей границе или готовая краевая матрица `robin` (дифференцированные ГУ, как в транзиенте).
    """
    K = assemble_stiffness_sparse(space, k)
    return (K + _robin_matrix(space, h, robin)).tocsr()


def runaway_threshold(space: LagrangeP1Space2D, k, h: float | None = None, *, robin=None) -> float:
    """
    Критическая чувствительность потерь s_crit = наименьшее обобщённое собств. значение
    (K+R)v = s·M·v (M — матрица масс) — порог для ОДНОРОДНОЙ обратной связи s ПО ВСЕЙ ОБЛАСТИ.
    При s ≥ s_crit система разгоняется. Для обратной связи, локализованной в меди (реальная машина), —
    `runaway_margin`: сравнивать s меди с этим s_crit нельзя.

    Ищется ОДНО значение сдвигом-обращением около нуля: A SPD ⇒ все s > 0, и ближайшее к
    нулю — наименьшее. Полный плотный спектр не нужен (на сетке машины это O(n³) и (n,n) в памяти).
    """
    A = thermal_operator(space, k, h, robin=robin)
    M = assemble_mass_sparse(space)
    w = eigsh(A, k=1, M=M, sigma=0.0, which="LM", return_eigenvectors=False)
    return float(w[0])


def runaway_margin(space: LagrangeP1Space2D, k, loss_sensitivity_cells, *, h: float | None = None,
                   robin=None) -> float:
    """
    Множитель запаса до теплового разгона λ_min — наименьшее собственное значение (K+R)v = λ·M_s·v,
    M_s,ij = ∫ s(x)·φ_i·φ_j dx, где s(x) ≥ 0 — чувствительность потерь по ячейкам [Вт/(м³·K)] (для меди
    `copper_loss_sensitivity(j_loss)`: ноль вне пазов). λ_min > 1 ⇒ стационар существует и устойчив;
    λ_min ≤ 1 ⇒ разгон; обратная связь, умноженная на λ_min, выводит ровно на порог. Граничные условия —
    те же, что в транзиенте: однородное `h` или готовая краевая матрица `robin` (например, `assemble_robin_multi`
    с h_in к раме и h_out к среде; температуры отсчёта на порог не влияют). Для однородного s по всей области
    λ_min = `runaway_threshold`/s.

    M_s вырождена вне источника (полуопределена), поэтому λ ищется сдвигом-обращением около нуля:
    (K+R)⁻¹M_s v = (1/λ)·v — наибольшее 1/λ даёт наименьшее λ, бесконечные λ (нуль-пространство M_s) не мешают.
    Без обратной связи (s ≡ 0) разгона нет: λ_min = ∞.
    """
    s = np.asarray(loss_sensitivity_cells, dtype=float).reshape(-1)
    if s.shape != (space.mesh.n_cells,):
        raise ValueError("loss_sensitivity_cells — по значению на ячейку (n_cells,).")
    if not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("чувствительность потерь s(x) должна быть конечной и неотрицательной.")
    if not np.any(s > 0.0):
        return float("inf")
    A = thermal_operator(space, k, h, robin=robin)
    Ms = assemble_mass_sparse(space, s)
    w = eigsh(A, k=1, M=Ms, sigma=0.0, which="LM", return_eigenvectors=False)
    return float(w[0])


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
    M = assemble_mass_sparse(space)
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
    lu = splu(A.tocsc())                  # A не меняется по итерациям ⇒ разложение одно
    M = assemble_mass_sparse(space)
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
        T_new = lu.solve(f0 + s * (M @ T))

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
