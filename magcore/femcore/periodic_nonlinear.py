from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.femcore.assembly import (
    assemble_magnetization_rhs,
    assemble_mixed_coulomb_system,
)
from magcore.femcore.periodic import PeriodicReduction, expand_solution, reduce_system
from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh import TetraMesh
from magcore.nonlinear.picard import resolve_magnetization, run_picard_fixed_point


@dataclass(frozen=True, slots=True)
class PeriodicPicardResult:
    a: np.ndarray
    p: np.ndarray
    B_cells: np.ndarray            # (n_cells,3) B=curl A
    H_cells: np.ndarray            # (n_cells,3) H=νB−νB_r
    nu_cells: np.ndarray           # (n_cells,) ν, на которой собрана финальная система
    nu_br_cells: np.ndarray        # (n_cells,3) финальный источник ν·B_r
    n_iterations: int
    converged: bool
    rel_change_history: tuple[float, ...]


def _zero_j(_x):
    return np.zeros(3)


def solve_periodic_nonlinear_mixed_picard(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    nu_of_B,
    *,
    nu_init,
    reduction: PeriodicReduction,
    dirichlet_reduced_dofs,
    j_fn=None,
    magnetization=None,
    max_iter: int = 80,
    tol: float = 1.0e-6,
    relaxation: float = 1.0,
    curl_quadrature_order: int = 1,
    coupling_quadrature_order: int = 2,
    rhs_quadrature_order: int = 3,
    magnetization_quadrature_order: int = 1,
) -> PeriodicPicardResult:
    """
    Нелинейный Picard для ОГРАНИЧЕННОЙ ПЕРИОДИЧЕСКОЙ смешанной A-Coulomb задачи
    (combined сталь+магнит на сегменте PMSM; §8.7). Аналог `solve_coupled_nonlinear_picard`,
    но открытая граница заменена на: периодическая редукция (`reduction`) + Dirichlet
    (`dirichlet_reduced_dofs` — индексы в РЕДУЦИРОВАННОЙ системе) на прочих гранях.

    Итерация: собрать `ν^k(|B^k|)` по ячейкам + источник намагниченности
    `(νB_r)^k(состояние)` → собрать ограниченную смешанную систему (с extra-RHS) →
    редуцировать (`TᵀMT`) → наложить Dirichlet → solve → expand → `B=curl A`,
    `H=νB−νB_r` → обновить ν. Критерий: ‖ΔB‖/‖B‖<tol. Под-релаксация ν.

    `magnetization`: None | статич. (n_cells,3) ν·B_r | callable(B,H,ν)->(n_cells,3)
    (магнит с состоянием/коленом — напр. `hybrid.magnet_demag.MagnetDemagPolicy`).
    """
    if vector_space.mesh is not mesh or scalar_space.mesh is not mesh:
        raise ValueError("vector_space/scalar_space must be built on the provided mesh.")
    if not callable(nu_of_B):
        raise ValueError("nu_of_B must be callable.")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1].")

    n_cells = mesh.n_cells
    nu_cells = np.asarray(nu_init, dtype=float).copy()
    if nu_cells.shape != (n_cells,):
        raise ValueError("nu_init must have shape (n_cells,).")
    nA = vector_space.ndofs
    j_eff = _zero_j if j_fn is None else j_fn
    dir_idx = np.asarray(list(dirichlet_reduced_dofs), dtype=int)

    # Источник намагниченности → функция состояния mag_fn(B,H,ν)->(n_cells,3) (общий helper).
    mag_fn = resolve_magnetization(magnetization, n_cells, dim=3)

    # Backend-шаг: заморозить источник намагниченности → собрать ограниченную смешанную
    # систему → редуцировать (TᵀMT) → Dirichlet → solve → expand → B=curl A, H=νB−νB_r.
    # Состояние (a,p,H,νB_r) в замыкании; общий цикл владеет релаксацией ν и сходимостью.
    state: dict[str, object] = {
        "a": np.zeros(nA, dtype=float),
        "p": np.zeros(scalar_space.ndofs, dtype=float),
        "B": np.zeros((n_cells, 3), dtype=float),
        "H": np.zeros((n_cells, 3), dtype=float),
        "nu_br": np.zeros((n_cells, 3), dtype=float),
    }

    def step(nu_frozen: np.ndarray) -> np.ndarray:
        nu_br_cells = np.asarray(mag_fn(state["B"], state["H"], nu_frozen), dtype=float)
        if nu_br_cells.shape != (n_cells, 3):
            raise ValueError("magnetization callable must return shape (n_cells, 3).")
        f_br = assemble_magnetization_rhs(
            mesh, vector_space, nu_br_cells, quadrature_order=magnetization_quadrature_order
        )
        M, b = assemble_mixed_coulomb_system(
            mesh, vector_space, scalar_space, nu=nu_frozen, J_fn=j_eff,
            extra_vector_rhs=f_br,
            curl_quadrature_order=curl_quadrature_order,
            coupling_quadrature_order=coupling_quadrature_order,
            rhs_quadrature_order=rhs_quadrature_order,
        )
        M_red, b_red = reduce_system(M, b, reduction)
        for r in dir_idx:
            M_red[r, :] = 0.0
            M_red[:, r] = 0.0
            M_red[r, r] = 1.0
            b_red[r] = 0.0

        try:
            x_red = np.linalg.solve(M_red, b_red)
        except np.linalg.LinAlgError:
            x_red, *_ = np.linalg.lstsq(M_red, b_red, rcond=None)

        x = expand_solution(x_red, reduction)
        a, p = x[:nA], x[nA:]
        B_cells = np.array(
            [evaluate_curl_on_cell(vector_space, a, c) for c in range(n_cells)], dtype=float
        )
        H_cells = nu_frozen[:, None] * B_cells - nu_br_cells
        state.update(a=a, p=p, B=B_cells, H=H_cells, nu_br=nu_br_cells)
        return B_cells

    loop = run_picard_fixed_point(
        nu_init=nu_cells,
        nu_of_B=nu_of_B,
        step=step,
        max_iter=max_iter,
        tol=tol,
        relaxation=relaxation,
    )

    return PeriodicPicardResult(
        a=state["a"], p=state["p"], B_cells=loop.B_cells, H_cells=state["H"],
        nu_cells=loop.nu_cells, nu_br_cells=state["nu_br"],
        n_iterations=loop.n_iterations, converged=loop.converged,
        rel_change_history=loop.rel_change_history,
    )
