from __future__ import annotations

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.magnet_demag import (
    MagnetDemagPolicy,
    compute_demag_risk_map,
)
from magcore.hybrid.nonlinear import CoupledPicardResult, solve_coupled_nonlinear_picard
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _run_knee(T: float, h0_phys_z: float = 0.0, n: int = 2, relaxation: float = 0.4):
    """Куб-магнит N42SH (весь куб) при T со встречным приложенным полем h0_phys_z [А/м]."""
    magnet = n42sh_magnet(easy_axis=[0.0, 0.0, 1.0])
    mesh = build_structured_unit_cube_tetra_mesh(n)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    nc = mesh.n_cells

    mask = np.ones(nc, dtype=bool)
    nu_mag = 1.0 / magnet.mu_rec
    policy = MagnetDemagPolicy(magnet, mask, T, nc, relaxation=relaxation)
    # Мост единиц: H0_solver[Тл] = μ₀·H0_phys[А/м].
    h0 = np.array([0.0, 0.0, MU0 * h0_phys_z]) if h0_phys_z else None

    res = solve_coupled_nonlinear_picard(
        ci, vs, ss,
        nu_of_B=lambda B: np.full(nc, nu_mag),
        nu_init=np.full(nc, nu_mag),
        magnetization=policy,
        applied_field_h0=h0,
        tol=1e-6, max_iter=80,
    )
    rmap = compute_demag_risk_map(magnet, res, mask, T)
    return magnet, res, rmap


# --------------------------------------------------------------------------------------
# (A) ФИЗИЧЕСКАЯ КОРРЕКТНОСТЬ: NdFeB при комнатной температуре — БЕЗОПАСЕН.
# Колено далеко (H_k≈1.35e6 А/м), саморазмагничивание куба ~−4e5 ⇒ потерь нет.
# --------------------------------------------------------------------------------------
def test_no_irreversible_demag_at_room_temperature() -> None:
    magnet, res, rmap = _run_knee(20.0)
    assert res.converged
    assert res.n_iterations <= 3  # выше колена B_r_eff постоянна ⇒ мгновенная сходимость
    assert rmap.n_demagnetized == 0
    assert np.all(rmap.margin > 0.0)
    assert np.allclose(rmap.Br_eff, rmap.Br_nominal)
    assert abs(rmap.total_loss) < 1e-9


# --------------------------------------------------------------------------------------
# (B) ЯДРО НОВИЗНЫ (2): ЧАСТИЧНОЕ необратимое размагничивание + RISK-MAP в связанной
# открытой задаче. Перегрев (80 °C) + встречное поле двигают ЧАСТЬ ячеек за колено;
# Picard по состоянию магнита сходится к рабочей точке на главной кривой; risk-map
# показывает пространственную структуру (часть ячеек размагничена, часть — нет).
# --------------------------------------------------------------------------------------
def test_partial_irreversible_demag_with_risk_map() -> None:
    magnet, res, rmap = _run_knee(80.0, h0_phys_z=-6.0e5)
    n_mag = rmap.cell_indices.size

    assert res.converged
    # Частичная картина: часть ячеек за коленом, часть — в безопасности.
    assert 0 < rmap.n_demagnetized < n_mag
    assert rmap.margin.min() < 0.0 < rmap.margin.max()

    # Необратимая потеря положительна, но НЕ катастрофична (магнит не коллапсирует).
    assert rmap.total_loss > 0.0
    assert np.all(rmap.Br_eff > 0.5 * rmap.Br_nominal)
    assert rmap.Br_eff.max() <= rmap.Br_nominal + 1e-9

    # Внутренняя согласованность карты (точные логические эквивалентности):
    #   demagnetized ⟺ margin<0 ⟺ Br_eff<Br_nom ⟺ loss>0.
    demag = rmap.demagnetized
    assert np.array_equal(demag, rmap.margin < 0.0)
    assert np.all(rmap.Br_eff[demag] < rmap.Br_nominal - 1e-12)
    assert np.allclose(rmap.Br_eff[~demag], rmap.Br_nominal)
    assert np.all(rmap.loss[demag] > 0.0)
    assert np.allclose(rmap.loss[~demag], 0.0)

    # Сильнее размагниченные ячейки (меньшая маржа) теряют больше ремнантности.
    order = np.argsort(rmap.margin)
    assert rmap.loss[order][0] >= rmap.loss[order][-1]  # худшая маржа ⇒ не меньшая потеря


# --------------------------------------------------------------------------------------
# (C) ЮНИТ: логика политики/risk-map на ЗАДАННОМ поле (без решателя) — детерминированно.
# Проверяем мост единиц (H_solver/μ₀), маржу к колену, монотонность потери по глубине.
# --------------------------------------------------------------------------------------
def test_risk_map_logic_on_prescribed_field() -> None:
    magnet = n42sh_magnet(easy_axis=[0.0, 0.0, 1.0])
    T = 80.0
    knee = magnet.knee_field(T)  # < 0 [А/м]

    # H_par: безопасно (выше колена), чуть и глубже за коленом (в пределах [колено, Hcj]).
    h_phys = np.array([0.5, 1.05, 1.15]) * knee  # [А/м], всё отрицательно
    n = h_phys.size
    H_cells = np.zeros((n, 3))
    H_cells[:, 2] = MU0 * h_phys  # H_solver = μ₀·H_phys

    res = CoupledPicardResult(
        a=np.zeros(0), p=np.zeros(0), psi=np.zeros(0), lam=np.zeros(0),
        B_cells=np.zeros((n, 3)), H_cells=H_cells,
        nu_cells=np.zeros(n), nu_br_cells=np.zeros((n, 3)),
        n_iterations=1, converged=True, residual_norm=0.0, rel_change_history=(),
    )
    rmap = compute_demag_risk_map(magnet, res, np.ones(n, dtype=bool), T)

    # Мост единиц восстанавливает физические А/м.
    assert np.allclose(rmap.H_par, h_phys)
    # Маржа: первая ячейка безопасна, две за коленом (всё глубже).
    assert rmap.margin[0] > 0.0
    assert rmap.margin[1] < 0.0
    assert rmap.margin[2] < rmap.margin[1] < 0.0
    assert np.array_equal(rmap.demagnetized, np.array([False, True, True]))
    # Потеря: 0 выше колена, растёт с глубиной за коленом.
    assert abs(rmap.loss[0]) < 1e-9
    assert rmap.loss[2] > rmap.loss[1] > 0.0
    assert rmap.Br_eff[2] < rmap.Br_eff[1] < rmap.Br_nominal
