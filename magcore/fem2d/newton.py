from __future__ import annotations

import numpy as np

from magcore.fem2d.assembly import (
    _scatter_local,
    assemble_current_rhs,
    assemble_current_rhs_piecewise,
    assemble_magnetization_rhs,
    assemble_stiffness_sparse,
    p1_cell_geometry,
)
from magcore.fem2d.nonlinear import Fem2DPicardResult
from magcore.fem2d.post import reconstruct_B_on_cells
from magcore.fem2d.solver import apply_dirichlet, solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.nonlinear.picard import resolve_magnetization

# Метод НЬЮТОНА для планарной магнитостатики −div(ν(|B|)∇A_z)=J_z+curl₂(νB_r).
# Касательная (Якобиан) использует ДИФФЕРЕНЦИАЛЬНУЮ релуктивность, поэтому сходимость
# КВАДРАТИЧНА и почти не зависит от сетки/релаксации (в отличие от хордового Пикара, где
# ω приходится подбирать под сетку). Глобализация — backtracking line-search. Робастная
# альтернатива `solve_nonlinear_2d_picard` без хрупкого параметра релаксации.
#
# Касательная на ячейке (P1, постоянные градиенты g_i=∇φ_i, ∇A=Σ A_i g_i):
#   T_ij = area·[ ν·(g_i·g_j) + 2·(dν/d|B|²)·(∇A·g_i)(∇A·g_j) ].
# Первый член = обычная жёсткость (хорда); второй = вклад насыщения (тот, что даёт Ньютон).
# Для воздуха/магнита ν=const ⇒ dν/d|B|²=0 ⇒ линейно (точно).


def assemble_newton_tangent(space, nu_cells, dnu_dB2_cells, a):
    """
    РАЗРЕЖЁННАЯ касательная T_ij = ∫ ν ∇φ_i·∇φ_j + 2·(dν/d|B|²)·(∇A·∇φ_i)(∇A·∇φ_j).
    Векторизовано + CSR (плотная (n,n) на реальных сетках не помещается в память).
    """
    nu = np.asarray(nu_cells, dtype=float)
    dnu = np.asarray(dnu_dB2_cells, dtype=float)
    cells, grad, area = p1_cell_geometry(space.mesh)      # (nc,3),(nc,3,2),(nc,)
    a_cell = np.asarray(a, dtype=float)[cells]            # (nc,3)
    gradA = np.einsum("ca,cad->cd", a_cell, grad)         # (nc,2) ∇A на ячейке
    w = np.einsum("cd,cad->ca", gradA, grad)              # (nc,3) w_a = ∇A·∇φ_a
    gg = np.einsum("cad,cbd->cab", grad, grad)            # (nc,3,3)
    ww = w[:, :, None] * w[:, None, :]                    # (nc,3,3)
    local = area[:, None, None] * (nu[:, None, None] * gg + 2.0 * dnu[:, None, None] * ww)
    return _scatter_local(cells, local, space.ndofs)


def solve_nonlinear_2d_newton(
    space: LagrangeP1Space2D,
    nu_and_dnu,
    *,
    nu_init,
    j_fn=None,
    j_cells=None,
    magnetization=None,
    dirichlet_dofs=None,
    dirichlet_values=0.0,
    max_iter: int = 40,
    tol: float = 1.0e-9,
    quadrature_order: int = 5,
) -> Fem2DPicardResult:
    """
    Ньютон для −div(ν(|B|)∇A_z)=J_z+curl₂(νB_r). `nu_and_dnu(B_cells:(n,2))` → (ν:(n,),
    dν/d|B|²:(n,)) — хордовая релуктивность и её производная по |B|² (для стали
    (ν_d−ν_chord)/(2|B|²); для воздуха/магнита 0). Источник магнита (демаг) обновляется на
    каждом шаге (мягкая нелинейность). Глобализация — backtracking по норме невязки.
    Возвращает тот же результат, что Пикар (совместим с risk-map и пр.).
    """
    if j_fn is not None and j_cells is not None:
        raise ValueError("задайте только один источник тока: j_fn ИЛИ j_cells.")
    nc = space.mesh.n_cells
    ddofs = np.asarray(space.boundary_dofs() if dirichlet_dofs is None else dirichlet_dofs, dtype=int)
    mag_fn = resolve_magnetization(magnetization, nc, dim=2)

    if j_fn is not None:
        f_cur = assemble_current_rhs(space, j_fn, quadrature_order=quadrature_order)
    elif j_cells is not None:
        f_cur = assemble_current_rhs_piecewise(space, j_cells)
    else:
        f_cur = np.zeros(space.ndofs, dtype=float)

    # Начальное A удовлетворяет Dirichlet (интерьер 0).
    a = np.zeros(space.ndofs, dtype=float)
    vals = np.asarray(dirichlet_values, dtype=float)
    a[ddofs] = vals if vals.ndim else float(vals)

    nu_br = np.zeros((nc, 2), dtype=float)

    def state(a_vec):
        """Поле, ν, источник, невязка при данном A (nu_br берётся замыканием — обновляется вне)."""
        B = reconstruct_B_on_cells(space, a_vec)
        nu, dnu = nu_and_dnu(B)
        nu = np.asarray(nu, dtype=float)
        H = nu[:, None] * B - nu_br
        f = f_cur + assemble_magnetization_rhs(space, nu_br)
        K = assemble_stiffness_sparse(space, nu)
        R = K @ a_vec - f
        R[ddofs] = 0.0
        return B, nu, np.asarray(dnu, dtype=float), H, R

    history: list[float] = []
    converged = False
    n_it = 0
    r0 = None
    B = reconstruct_B_on_cells(space, a)
    for k in range(max_iter):
        n_it = k + 1
        # обновить источник магнита по текущему полю (демаг), затем невязку+касательную
        nu0, _ = nu_and_dnu(B)
        H0 = np.asarray(nu0, dtype=float)[:, None] * B - nu_br
        nu_br = np.asarray(mag_fn(B, H0, np.asarray(nu0, dtype=float)), dtype=float)
        B, nu, dnu, H, R = state(a)
        rnorm = float(np.linalg.norm(R))
        history.append(rnorm)
        if r0 is None:
            r0 = max(rnorm, 1e-30)
        if rnorm <= tol * r0:            # ОТНОСИТЕЛЬНЫЙ критерий (масштабо-независим)
            converged = True
            break
        T = assemble_newton_tangent(space, nu, dnu, a)
        T_bc, rhs = apply_dirichlet(T, -R, ddofs, 0.0)
        da = solve_scalar(T_bc, rhs)
        # backtracking line-search по норме невязки
        alpha = 1.0
        for _ in range(12):
            _, _, _, _, Rt = state(a + alpha * da)
            if np.linalg.norm(Rt) < rnorm:
                break
            alpha *= 0.5
        a = a + alpha * da
        B = reconstruct_B_on_cells(space, a)
        if float(np.linalg.norm(alpha * da)) <= tol * (1.0 + float(np.linalg.norm(a))):
            converged = True
            break

    B = reconstruct_B_on_cells(space, a)
    nu, _ = nu_and_dnu(B)
    nu = np.asarray(nu, dtype=float)
    H = nu[:, None] * B - nu_br
    return Fem2DPicardResult(
        a=a, B_cells=B, H_cells=H, nu_cells=nu, nu_br_cells=nu_br,
        n_iterations=n_it, converged=converged, rel_change_history=tuple(history),
    )
