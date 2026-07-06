from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.femcore.assembly import assemble_magnetization_rhs
from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import (
    assemble_coupled_block_system,
    prepare_coupled_exterior,
)
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.solver import solve_coupled_block_system


@dataclass(frozen=True, slots=True)
class CoupledPicardResult:
    a: np.ndarray                 # Неделек-дофы A
    p: np.ndarray                 # множитель калибровки
    psi: np.ndarray               # внешний след ψ
    lam: np.ndarray               # внешний поток λ
    B_cells: np.ndarray           # (n_cells, 3) — B = curl A на ячейку
    H_cells: np.ndarray           # (n_cells, 3) — H = νB − νB_r на ячейку
    nu_cells: np.ndarray          # (n_cells,) — самосогласованная ν(|B|), на которой собран финал
    nu_br_cells: np.ndarray       # (n_cells, 3) — финальный источник ν·B_r (после состояния магнита)
    n_iterations: int
    converged: bool
    residual_norm: float          # ‖Mx−b‖ финальной линейной системы
    rel_change_history: tuple[float, ...]


def _zero_j(_x):
    return np.zeros(3)


def solve_coupled_nonlinear_picard(
    interface: CouplingInterface,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
    nu_of_B,
    *,
    nu_init,
    j_fn=None,
    magnetization=None,
    applied_field_h0: np.ndarray | None = None,
    mu0: float = 1.0,
    max_iter: int = 50,
    tol: float = 1.0e-6,
    relaxation: float = 1.0,
    config=None,
    curl_quadrature_order: int = 1,
    coupling_quadrature_order: int = 2,
    rhs_quadrature_order: int = 3,
    magnetization_quadrature_order: int = 1,
) -> CoupledPicardResult:
    """
    Нелинейный Picard (хордовый) для СВЯЗАННОЙ FEM/BEM A-Coulomb задачи в открытой
    области — общий движок «магнит ↔ нелинейное железо». См. docs/math/nonlinear_materials.md §8.

    Итерация k: заморозить поячеечную ν^k(|B^k|) И источник намагниченности
    (νB_r)^k(состояние) → собрать связанную блок-систему → решить → B^{k+1}=curl A →
    H^{k+1}=ν B − νB_r → обновить ν (и магнит). Критерий: ‖B^{k+1}−B^k‖/‖B^k‖<tol.

    Параметры
    ---------
    nu_of_B : callable(B_cells:(n_cells,3)) -> (n_cells,)
        Хордовая релуктивность по ячейке: воздух=const, сталь=SteelBHCurve.nu_chord(|B|),
        магнит=1/μ_rec (обычно const). Объединяет ВСЕ материальные области.
    nu_init : (n_cells,)  стартовая ν.
    j_fn : callable(x)->(3,) свободный ток (по умолчанию ноль).
    magnetization :
        None — магнита нет; ИЛИ статический массив (n_cells,3) значений ν·B_r
        (ненулевой только в ячейках магнита); ИЛИ callable(B_cells,H_cells,nu_cells)
        -> (n_cells,3) для МАГНИТА С СОСТОЯНИЕМ (колено/возврат: B_r_eff(H_min)).
        В первой итерации состояние-callable вызывается при B=H=0 (номинальный магнит).
    applied_field_h0 : (3,) однородное внешнее поле H₀ (опц.).

    Внешний BEM-блок при нелинейности линеен (зависит только от mu0) — между итерациями
    меняются лишь внутренний ν-блок и RHS намагниченности; экстерьер собирается заново
    для простоты (можно кэшировать позже).
    """
    mesh = interface.tetra_mesh
    if vector_space.mesh is not mesh or scalar_space.mesh is not mesh:
        raise ValueError("vector_space/scalar_space must be built on interface.tetra_mesh.")
    if not callable(nu_of_B):
        raise ValueError("nu_of_B must be callable.")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1].")

    n_cells = mesh.n_cells
    nu_cells = np.asarray(nu_init, dtype=float).copy()
    if nu_cells.shape != (n_cells,):
        raise ValueError("nu_init must have shape (n_cells,).")

    j_eff = _zero_j if j_fn is None else j_fn

    # Разрешаем источник намагниченности в функцию состояния mag_fn(B,H,nu)->(n_cells,3).
    if magnetization is None:
        def mag_fn(_B, _H, _nu):
            return np.zeros((n_cells, 3), dtype=float)
    elif callable(magnetization):
        mag_fn = magnetization
    else:
        static_mag = np.asarray(magnetization, dtype=float)
        if static_mag.shape != (n_cells, 3):
            raise ValueError("static magnetization must have shape (n_cells, 3).")
        def mag_fn(_B, _H, _nu):
            return static_mag

    B_cells = np.zeros((n_cells, 3), dtype=float)
    H_cells = np.zeros((n_cells, 3), dtype=float)
    nu_br_cells = np.zeros((n_cells, 3), dtype=float)
    a = np.zeros(vector_space.ndofs, dtype=float)
    p = np.zeros(scalar_space.ndofs, dtype=float)
    psi = np.zeros(interface.n_phi_dofs, dtype=float)
    lam = np.zeros(interface.n_flux_dofs, dtype=float)
    residual_norm = float("nan")

    # Линейный BEM-экстерьер + B_FΓ не зависят от ν/намагниченности ⇒ собираем ОДИН РАЗ
    # и переиспользуем на всех итерациях (точная оптимизация, не приближение).
    exterior_cache = prepare_coupled_exterior(
        interface, vector_space, mu0=mu0, config=config
    )

    B_prev: np.ndarray | None = None
    history: list[float] = []
    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        nu_br_cells = np.asarray(mag_fn(B_cells, H_cells, nu_cells), dtype=float)
        if nu_br_cells.shape != (n_cells, 3):
            raise ValueError("magnetization callable must return shape (n_cells, 3).")

        f_br = assemble_magnetization_rhs(
            mesh, vector_space, nu_br_cells,
            quadrature_order=magnetization_quadrature_order,
        )
        coupled = assemble_coupled_block_system(
            interface, vector_space, scalar_space,
            nu=nu_cells, j_fn=j_eff, extra_vector_rhs=f_br,
            applied_field_h0=applied_field_h0, mu0=mu0, config=config,
            exterior=exterior_cache,
            curl_quadrature_order=curl_quadrature_order,
            coupling_quadrature_order=coupling_quadrature_order,
            rhs_quadrature_order=rhs_quadrature_order,
        )
        sol = solve_coupled_block_system(coupled, scalar_space)
        a, p, psi, lam = sol.a, sol.p, sol.psi, sol.lam
        residual_norm = sol.residual_norm

        B_cells = np.array(
            [evaluate_curl_on_cell(vector_space, a, c) for c in range(n_cells)],
            dtype=float,
        )
        # H = ν(B − B_r) = ν B − (νB_r): источник νB_r уже поячеечный.
        H_cells = nu_cells[:, None] * B_cells - nu_br_cells

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

    return CoupledPicardResult(
        a=a, p=p, psi=psi, lam=lam,
        B_cells=B_cells, H_cells=H_cells,
        nu_cells=nu_cells, nu_br_cells=nu_br_cells,
        n_iterations=it, converged=converged,
        residual_norm=residual_norm,
        rel_change_history=tuple(history),
    )
