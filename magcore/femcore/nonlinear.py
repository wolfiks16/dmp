from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.femcore.assembly import assemble_mixed_coulomb_system
from magcore.femcore.boundary_conditions import (
    apply_zero_mixed_dirichlet_bc,
    find_mixed_boundary_dofs,
)
from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.solver import solve_mixed_coulomb_problem, split_mixed_solution
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh import TetraMesh


@dataclass(frozen=True, slots=True)
class PicardResult:
    a: np.ndarray                      # коэффициенты A_h
    p: np.ndarray                      # множитель Кулоновой калибровки
    B_cells: np.ndarray               # (n_cells, 3) — B=curl A на ячейку
    nu_cells: np.ndarray              # (n_cells,) — самосогласованная хордовая ν(|B|)
    n_iterations: int
    converged: bool
    rel_change_history: tuple[float, ...]


def solve_nonlinear_mixed_picard(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    nu_of_B,
    J_fn,
    *,
    nu_init,
    max_iter: int = 50,
    tol: float = 1.0e-6,
    relaxation: float = 1.0,
    curl_quadrature_order: int = 1,
    coupling_quadrature_order: int = 2,
    rhs_quadrature_order: int = 3,
) -> PicardResult:
    """
    Picard (хордовый) фиксированной точки для НЕЛИНЕЙНОЙ изотропной смешанной
    A-Coulomb задачи (нелинейная сталь). См. docs/math/nonlinear_materials.md §5.

    Шаг k: заморозить ν^k(|B^k|) по ячейкам → собрать линейную смешанную систему →
    наложить BC (n×A=0) → решить → B^{k+1}=curl A^{k+1} → обновить ν.
    Критерий сходимости: ‖B^{k+1}−B^k‖/‖B^k‖ < tol. Под-релаксация ν (relaxation∈(0,1]).

    Параметры:
      nu_of_B : callable(B_cells:(n_cells,3)) -> (n_cells,) — хордовая релуктивность
                по ячейке как функция |B| (сталь: SteelBHCurve.nu_chord; воздух: const).
      nu_init : (n_cells,) стартовая релуктивность.
    Только внутренний блок A_FF пересобирается между итерациями (внешний/связующий
    блок при нелинейности линеен — здесь решается FEM-only ограниченная задача).
    """
    if vector_space.mesh is not mesh or scalar_space.mesh is not mesh:
        raise ValueError("vector_space and scalar_space must be built on the provided mesh.")
    if not callable(nu_of_B):
        raise ValueError("nu_of_B must be callable.")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1].")

    n_cells = mesh.n_cells
    nu_cells = np.asarray(nu_init, dtype=float).copy()
    if nu_cells.shape != (n_cells,):
        raise ValueError("nu_init must have shape (n_cells,).")

    vector_bnd, scalar_bnd = find_mixed_boundary_dofs(
        vector_space=vector_space, scalar_space=scalar_space
    )
    nA = vector_space.ndofs

    B_prev: np.ndarray | None = None
    history: list[float] = []
    converged = False
    a = np.zeros(nA, dtype=float)
    p = np.zeros(scalar_space.ndofs, dtype=float)
    B_cells = np.zeros((n_cells, 3), dtype=float)
    it = 0

    for it in range(1, max_iter + 1):
        system_matrix, rhs = assemble_mixed_coulomb_system(
            mesh=mesh,
            vector_space=vector_space,
            scalar_space=scalar_space,
            nu=nu_cells,
            J_fn=J_fn,
            curl_quadrature_order=curl_quadrature_order,
            coupling_quadrature_order=coupling_quadrature_order,
            rhs_quadrature_order=rhs_quadrature_order,
        )
        A_bc, b_bc = apply_zero_mixed_dirichlet_bc(
            A=system_matrix,
            b=rhs,
            vector_dofs=vector_bnd,
            scalar_dofs=scalar_bnd,
            n_vector_dofs=nA,
        )
        x = solve_mixed_coulomb_problem(A_bc, b_bc)
        a, p = split_mixed_solution(x, nA)
        B_cells = np.array(
            [evaluate_curl_on_cell(vector_space, a, c) for c in range(n_cells)],
            dtype=float,
        )

        if B_prev is not None:
            denom = float(np.linalg.norm(B_prev))
            rel = float(np.linalg.norm(B_cells - B_prev)) / max(denom, 1.0e-30)
            history.append(rel)
            if rel < tol:
                converged = True
                break
        B_prev = B_cells

        nu_new = np.asarray(nu_of_B(B_cells), dtype=float)
        if nu_new.shape != (n_cells,):
            raise ValueError("nu_of_B must return an array of shape (n_cells,).")
        nu_cells = (1.0 - relaxation) * nu_cells + relaxation * nu_new

    # Самосогласованная релуктивность при найденном B (для отчёта/диагностики).
    nu_self = np.asarray(nu_of_B(B_cells), dtype=float)

    return PicardResult(
        a=a,
        p=p,
        B_cells=B_cells,
        nu_cells=nu_self,
        n_iterations=it,
        converged=converged,
        rel_change_history=tuple(history),
    )
