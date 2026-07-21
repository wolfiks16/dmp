from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.fem2d.assembly import (
    assemble_current_rhs,
    assemble_current_rhs_piecewise,
    assemble_magnetization_rhs,
    assemble_stiffness_sparse,
)
from magcore.fem2d.post import reconstruct_B_on_cells
from magcore.fem2d.solver import apply_dirichlet, solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.nonlinear.picard import resolve_magnetization, run_picard_fixed_point


@dataclass(frozen=True, slots=True)
class Fem2DPicardResult:
    a: np.ndarray                 # (ndofs,) узловой A_z
    B_cells: np.ndarray           # (n_cells, 2) — B=(∂_yA_z, −∂_xA_z)
    H_cells: np.ndarray           # (n_cells, 2) — H=νB − νB_r
    nu_cells: np.ndarray          # (n_cells,) — ν финальной сборки
    nu_br_cells: np.ndarray       # (n_cells, 2) — финальный источник ν·B_r
    n_iterations: int
    converged: bool
    rel_change_history: tuple[float, ...]


def solve_nonlinear_2d_picard(
    space: LagrangeP1Space2D,
    nu_of_B,
    *,
    nu_init,
    j_fn=None,
    j_cells=None,
    magnetization=None,
    dirichlet_dofs=None,
    dirichlet_values=0.0,
    max_iter: int = 50,
    tol: float = 1.0e-6,
    relaxation: float = 1.0,
    quadrature_order: int = 5,
    warm_start: "Fem2DPicardResult | None" = None,
) -> Fem2DPicardResult:
    """
    Нелинейный Picard для планарной задачи −div(ν(|B|)∇A_z)=J_z+curl₂(νB_r) на P1.

    Переиспользует РАЗМЕРНО-НЕЗАВИСИМОЕ ядро `run_picard_fixed_point` (2D-0): цикл
    владеет релаксацией ν и критерием сходимости, а backend-шаг здесь собирает
    планарную систему (жёсткость + ток + магнит), накладывает Dirichlet и решает.
    Демонстрирует общность ядра между 3D- и 2D-backend'ами.

    nu_of_B : callable(B_cells:(n_cells,2)) -> (n_cells,)  хордовая ν(|B|) (воздух/сталь/магнит).
    j_fn : callable(x:(2,))->float  внеплоскостной ток (RHS собирается ОДИН раз, ν-независим).
    j_cells : (n_cells,) КУСОЧНО-ПОСТОЯННЫЙ ток по ячейкам (альтернатива j_fn, для обмотки
              из P3; ровно один из j_fn/j_cells). RHS в единицах решателя (масштаб μ₀ — на
              вызывающем; см. machines/static_solver).
    magnetization : None | (n_cells,2) ν·B_r | callable(B,H,ν)->(n_cells,2) (магнит с коленом).
    dirichlet_dofs : узлы Dirichlet (по умолчанию — граница сетки).
    warm_start : предыдущий `Fem2DPicardResult` как НАЧАЛЬНОЕ приближение (B,H для оценки
        состояния материалов на 1-й итерации). Неподвижная точка и критерий сходимости не
        меняются — только стартовая точка, поэтому ответ тот же с точностью до `tol`.
        Ключевое для расчёта во времени: поле между шагами меняется слабо ⇒ 2–3 итерации
        вместо десятков.
    """
    if j_fn is not None and j_cells is not None:
        raise ValueError("задайте только один источник тока: j_fn ИЛИ j_cells.")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1].")
    n_cells = space.mesh.n_cells
    nu0 = np.asarray(nu_init, dtype=float).copy()
    if nu0.shape != (n_cells,):
        raise ValueError("nu_init must have shape (n_cells,).")

    ddofs = space.boundary_dofs() if dirichlet_dofs is None else dirichlet_dofs
    mag_fn = resolve_magnetization(magnetization, n_cells, dim=2)

    # Токовый RHS линеен и ν-независим ⇒ собираем один раз (непрерывный j_fn или
    # кусочно-постоянный j_cells обмотки).
    if j_fn is not None:
        f_current = assemble_current_rhs(space, j_fn, quadrature_order=quadrature_order)
    elif j_cells is not None:
        f_current = assemble_current_rhs_piecewise(space, j_cells)
    else:
        f_current = np.zeros(space.ndofs, dtype=float)

    state: dict[str, object] = {
        "a": np.zeros(space.ndofs, dtype=float),
        "B": np.zeros((n_cells, 2), dtype=float),
        "H": np.zeros((n_cells, 2), dtype=float),
        "nu_br": np.zeros((n_cells, 2), dtype=float),
    }
    if warm_start is not None:
        state.update(a=np.asarray(warm_start.a, dtype=float).copy(),
                     B=np.asarray(warm_start.B_cells, dtype=float).copy(),
                     H=np.asarray(warm_start.H_cells, dtype=float).copy())
        nu0 = np.asarray(warm_start.nu_cells, dtype=float).copy()

    def step(nu_frozen: np.ndarray) -> np.ndarray:
        nu_br = np.asarray(mag_fn(state["B"], state["H"], nu_frozen), dtype=float)
        if nu_br.shape != (n_cells, 2):
            raise ValueError("magnetization callable must return shape (n_cells, 2).")
        K = assemble_stiffness_sparse(space, nu_frozen)
        f = f_current + assemble_magnetization_rhs(space, nu_br)
        K_bc, f_bc = apply_dirichlet(K, f, ddofs, dirichlet_values)
        a = solve_scalar(K_bc, f_bc)
        B = reconstruct_B_on_cells(space, a)
        H = nu_frozen[:, None] * B - nu_br
        state.update(a=a, B=B, H=H, nu_br=nu_br)
        return B

    loop = run_picard_fixed_point(
        nu_init=nu0,
        nu_of_B=nu_of_B,
        step=step,
        max_iter=max_iter,
        tol=tol,
        relaxation=relaxation,
    )

    return Fem2DPicardResult(
        a=state["a"],
        B_cells=loop.B_cells,
        H_cells=state["H"],
        nu_cells=loop.nu_cells,
        nu_br_cells=state["nu_br"],
        n_iterations=loop.n_iterations,
        converged=loop.converged,
        rel_change_history=loop.rel_change_history,
    )
