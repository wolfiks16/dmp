from __future__ import annotations

import numpy as np

from magcore.cancel import check as cancel_check
from magcore.fem2d.assembly import (
    _scatter_local,
    assemble_current_rhs,
    assemble_current_rhs_piecewise,
    p1_cell_geometry,
)
from magcore.fem2d.nonlinear import Fem2DPicardResult
from magcore.fem2d.solver import apply_dirichlet, solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.nonlinear.picard import resolve_magnetization

# Метод НЬЮТОНА для планарной магнитостатики. Слабая форма в общем виде:
#     R_i(A) = ∫ H(B) · rot φ_i dx − ∫ J φ_i dx = 0,   B = rot A = (∂_yA, −∂_xA),
# касательная  T_ij = ∫ rot φ_i · (dH/dB) · rot φ_j dx.
# Это охватывает ОБА закона материала:
#   * изотропный (воздух, линейный, сталь):  H = ν(|B|)·B − ν·B_r,  dH/dB = ν·I + 2·(dν/d|B|²)·B⊗B —
#     для стали касательная даёт квадратичную сходимость, не зависящую от сетки;
#   * анизотропный магнит (`magnet_law`): вдоль лёгкой оси — рабочая ветвь кривой размагничивания,
#     поперёк — μ⊥; H и касательная берутся у одной ветви. Без него магнит за коленом приходится
#     считать внешним циклом по источнику с релаксацией, а он там не сжимает (Л-93).
# Глобализация — backtracking line-search по норме невязки. «Сошлось» ставится ТОЛЬКО по невязке:
# исчерпанный линейный поиск (шаг стал крошечным при большой невязке) — застой, а не сходимость (Л-108).


def _rot_basis(space: LagrangeP1Space2D):
    """Ячейки, rot φ_a = (∂_yφ_a, −∂_xφ_a) и площади: в этом базисе B = Σ_a A_a·rot φ_a."""
    cells, grad, area = p1_cell_geometry(space.mesh)           # (nc,3),(nc,3,2),(nc,)
    return cells, np.stack([grad[:, :, 1], -grad[:, :, 0]], axis=2), area


def _b_on_cells(cells, rot, a) -> np.ndarray:
    """Поячеечное B = rot A (то же, что `post.reconstruct_B_on_cells`, но векторизовано)."""
    return np.einsum("ca,cad->cd", np.asarray(a, dtype=float)[cells], rot)


def assemble_tangent_2d(space: LagrangeP1Space2D, d_cells: np.ndarray):
    """РАЗРЕЖЁННАЯ касательная T_ij = ∫ rot φ_i·D·rot φ_j; D (n_cells,2,2) — поячеечная dH/dB."""
    cells, rot, area = _rot_basis(space)
    d = np.asarray(d_cells, dtype=float)
    if d.shape != (space.mesh.n_cells, 2, 2):
        raise ValueError("d_cells must have shape (n_cells, 2, 2).")
    local = area[:, None, None] * np.einsum("cai,cij,cbj->cab", rot, d, rot)
    return _scatter_local(cells, local, space.ndofs)


def assemble_newton_tangent(space, nu_cells, dnu_dB2_cells, a):
    """
    Касательная изотропного закона: T_ij = ∫ ν ∇φ_i·∇φ_j + 2·(dν/d|B|²)·(∇A·∇φ_i)(∇A·∇φ_j).
    Первый член — обычная жёсткость (хорда), второй — вклад насыщения. Это `assemble_tangent_2d`
    с D = ν·I + 2·(dν/d|B|²)·B⊗B (∇A·∇φ_a = rot A·rot φ_a).
    """
    cells, rot, _ = _rot_basis(space)
    B = _b_on_cells(cells, rot, a)
    nu = np.asarray(nu_cells, dtype=float)
    dnu = np.asarray(dnu_dB2_cells, dtype=float)
    eye = np.broadcast_to(np.eye(2), (space.mesh.n_cells, 2, 2))
    return assemble_tangent_2d(space, nu[:, None, None] * eye + 2.0 * dnu[:, None, None] * B[:, :, None] * B[:, None, :])


def solve_nonlinear_2d_newton(
    space: LagrangeP1Space2D,
    nu_and_dnu,
    *,
    nu_init,
    j_fn=None,
    j_cells=None,
    magnetization=None,
    magnet_law=None,
    dirichlet_dofs=None,
    dirichlet_values=0.0,
    max_iter: int = 40,
    tol: float = 1.0e-9,
    quadrature_order: int = 5,
) -> Fem2DPicardResult:
    """
    Ньютон для −div(ν(|B|)∇A_z) = J_z + curl₂(νB_r). `nu_and_dnu(B_cells:(n,2))` → (ν:(n,),
    dν/d|B|²:(n,)) — хордовая релуктивность и её производная по |B|² (для стали
    (ν_d−ν_chord)/(2|B|²); для воздуха/магнита 0).

    Магнит задаётся ОДНИМ из двух способов:
      * `magnet_law` (рекомендуется) — закон магнита в касательной (`fem2d.magnet_law.MagnetLaw2D`):
        отдельного источника нет, релаксации нет, за коленом сходится так же быстро, как до него;
      * `magnetization` — источник ν·B_r, обновляемый на каждом шаге (совместимость; за коленом
        итерация по источнику может не сходиться — см. Л-93).

    Возвращает тот же результат, что Пикар (совместим с картой риска и пост-процессингом);
    `converged` ставится только по относительной невязке.
    """
    if j_fn is not None and j_cells is not None:
        raise ValueError("задайте только один источник тока: j_fn ИЛИ j_cells.")
    if magnetization is not None and magnet_law is not None:
        raise ValueError("магнит задаётся ЛИБО законом (magnet_law), ЛИБО источником (magnetization).")
    nc = space.mesh.n_cells
    ddofs = np.asarray(space.boundary_dofs() if dirichlet_dofs is None else dirichlet_dofs, dtype=int)
    mag_fn = resolve_magnetization(magnetization, nc, dim=2)
    cells, rot, area = _rot_basis(space)
    eye = np.broadcast_to(np.eye(2), (nc, 2, 2))

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
        """Поле, ν, касательная, H и невязка при данном A (nu_br берётся замыканием — обновляется вне)."""
        B = _b_on_cells(cells, rot, a_vec)
        nu, dnu = nu_and_dnu(B)
        nu = np.asarray(nu, dtype=float)
        dnu = np.asarray(dnu, dtype=float)
        H = nu[:, None] * B - nu_br
        D = nu[:, None, None] * eye + 2.0 * dnu[:, None, None] * B[:, :, None] * B[:, None, :]
        if magnet_law is not None:
            H_mag, D_mag = magnet_law(B)
            H[magnet_law.idx] = H_mag
            D[magnet_law.idx] = D_mag
        local = area[:, None] * np.einsum("cai,ci->ca", rot, H)
        R = np.bincount(cells.ravel(), weights=local.ravel(), minlength=space.ndofs) - f_cur
        R[ddofs] = 0.0
        return B, nu, dnu, H, D, R

    history: list[float] = []
    converged = False
    n_it = 0
    r0 = None
    B = _b_on_cells(cells, rot, a)
    for k in range(max_iter):
        cancel_check()                               # отмена расчёта — между итерациями
        n_it = k + 1
        if magnet_law is None:                       # источник магнита по текущему полю (мягкая нелинейность)
            nu0, _ = nu_and_dnu(B)
            H0 = np.asarray(nu0, dtype=float)[:, None] * B - nu_br
            nu_br = np.asarray(mag_fn(B, H0, np.asarray(nu0, dtype=float)), dtype=float)
        B, nu, dnu, H, D, R = state(a)
        rnorm = float(np.linalg.norm(R))
        history.append(rnorm)
        if r0 is None:
            r0 = max(rnorm, 1e-30)
        if rnorm <= tol * r0:            # ОТНОСИТЕЛЬНЫЙ критерий (масштабо-независим)
            converged = True
            break
        T = assemble_tangent_2d(space, D)
        T_bc, rhs = apply_dirichlet(T, -R, ddofs, 0.0)
        da = solve_scalar(T_bc, rhs)
        # backtracking line-search по норме невязки
        alpha = 1.0
        improved = False
        for _ in range(12):
            *_, Rt = state(a + alpha * da)
            if np.linalg.norm(Rt) < rnorm:
                improved = True
                break
            alpha *= 0.5
        a = a + alpha * da
        B = _b_on_cells(cells, rot, a)
        # Линейный поиск не смог уменьшить невязку — это ЗАСТОЙ, дальше идти некуда. Выходим, но
        # «сошлось» ставит только невязка ВОЗВРАЩАЕМОГО решения (считается ниже), а не малость шага (Л-108).
        if not improved:
            break

    B, nu, _, H, _, R = state(a)
    if magnet_law is not None:
        # Закон в касательной состояния не хранит, поэтому невязку считаем на ВОЗВРАЩАЕМОМ A: последний
        # шаг мог уже привести решение к нулю, и объявлять «не сошлось» по невязке до шага нечестно.
        # (Путь с ИСТОЧНИКОМ решает по циклу: там невязка меряется сразу после обновления источника и
        # уже учитывает его несогласованность, а повторный вызов сдвинул бы источник ещё на шаг релаксации.)
        rnorm = float(np.linalg.norm(R))
        if r0 is None:
            r0 = max(rnorm, 1e-30)
        if not history or rnorm != history[-1]:
            history.append(rnorm)
        converged = rnorm <= tol * r0
    if magnet_law is not None:
        # H в ячейках магнита уже по закону ветви (см. `state`); эквивалентный источник ν·B_r той же
        # ветви сохраняет тождество H = ν·B − ν·B_r для пост-обработки.
        nu_br[magnet_law.idx] = nu[magnet_law.idx, None] * B[magnet_law.idx] - H[magnet_law.idx]
    return Fem2DPicardResult(
        a=a, B_cells=B, H_cells=H, nu_cells=nu, nu_br_cells=nu_br,
        n_iterations=n_it, converged=converged, rel_change_history=tuple(history),
    )
