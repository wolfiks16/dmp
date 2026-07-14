from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.excitation import winding_current_density
from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry
from magcore.fem2d.machines.static_solver import MachineStaticResult, machine_reluctivity
from magcore.fem2d.machines.winding import WindingLayout
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.hybrid.magnet_demag import (
    DemagRiskMap,
    MagnetDemagPolicy,
    compute_demag_risk_map,
)

# P5: КОЛИЧЕСТВЕННАЯ необратимая потеря — не «за коленом/нет», а СКОЛЬКО теряет магнит и
# машина. Два уровня: (1) агрегаты по магниту из risk-map (доля площади за коленом, средняя/
# макс. относит. потеря Br, потерянный «магнитный поток источника»); (2) машинный итог —
# падение потокосцепления холостого хода (= падение ЭДС/моментной постоянной) ПОСЛЕ демага.
#
# Механика (2): под worst-case нагрузкой защёлкиваем необратимое состояние (track_worst_point:
# H_min латчится по ячейкам), затем считаем λ холостого хода с ТЕМ ЖЕ состоянием (ток снят,
# Br_eff остаётся пониженной — необратимость) и сравниваем с номинальным λ при той же T.


@dataclass(frozen=True, slots=True)
class MagnetLossAggregate:
    """Агрегаты необратимой потери по объёму магнита (из risk-map)."""

    demag_area_fraction: float          # доля площади магнита за коленом (m<0)
    mean_loss_frac: float               # средняя относит. потеря Br (площадно-взвеш.)
    max_loss_frac: float                # макс. относит. потеря Br по ячейке
    total_lost_br_area: float           # Σ ΔBr·area [Тл·м²] (×L = потеря потока источника)


def magnet_loss_aggregate(
    geometry: MachineGeometry, risk: DemagRiskMap
) -> MagnetLossAggregate:
    """Свести поячеечную потерю risk-map в агрегаты по магниту (площадно-взвешенные)."""
    idx = risk.cell_indices
    areas = np.array([geometry.mesh.cell_area(int(c)) for c in idx], dtype=float)
    a_tot = float(areas.sum())
    br_nom = risk.Br_nominal
    demag_area = float(areas[risk.demagnetized].sum())
    if a_tot <= 0.0 or br_nom <= 0.0:
        return MagnetLossAggregate(0.0, 0.0, 0.0, 0.0)
    return MagnetLossAggregate(
        demag_area_fraction=demag_area / a_tot,
        mean_loss_frac=float((risk.loss * areas).sum() / (br_nom * a_tot)),
        max_loss_frac=float(risk.loss.max() / br_nom),
        total_lost_br_area=float((risk.loss * areas).sum()),
    )


def phase_flux_linkage(
    geometry: MachineGeometry,
    layout: WindingLayout,
    a: np.ndarray,
    *,
    turns_per_slot: float,
    axial_length: float | None = None,
) -> np.ndarray:
    """
    Потокосцепление фаз [Вб] из решённого A_z (2D): λ_ph = L·Σ_{s∈ph} sign_s·N·⟨A_z⟩_s,
    где ⟨A_z⟩_s — площадно-взвешенное среднее A_z по пазу s (∫(N/A_slot)A_z dA = N·⟨A_z⟩).
    N = turns_per_slot. Возвращает (3,) для фаз A/B/C. Основа ЭДС/момента (P6).
    """
    mesh = geometry.mesh
    L = geometry.params.axial_length if axial_length is None else float(axial_length)
    N = float(turns_per_slot)
    sid = geometry.slot_id
    lam = np.zeros(3, dtype=float)
    for s in range(layout.n_slots):
        cells = np.where(sid == s)[0]
        if cells.size == 0:
            continue
        areas = np.array([mesh.cell_area(int(c)) for c in cells], dtype=float)
        cell_mean_az = a[mesh.cells[cells]].mean(axis=1)          # (ncell,) среднее по узлам
        aw_mean = float((cell_mean_az * areas).sum() / areas.sum())
        lam[layout.phase_of_slot[s]] += layout.sign_of_slot[s] * N * aw_mean
    return L * lam


def _clarke_amplitude(vec3: np.ndarray) -> float:
    """Амплитуда пространственного вектора из фазных величин (Кларк, инвариант амплитуды)."""
    alpha = vec3[0] - 0.5 * vec3[1] - 0.5 * vec3[2]
    beta = (np.sqrt(3.0) / 2.0) * (vec3[1] - vec3[2])
    return float(np.hypot(alpha, beta))


@dataclass(frozen=True, slots=True)
class DemagImpact:
    """Машинный итог демага: падение потокосцепления ХХ (= падение ЭДС/момента) + агрегаты."""

    risk: DemagRiskMap                  # карта риска под worst-case нагрузкой
    aggregate: MagnetLossAggregate      # агрегаты потери по магниту
    lam_nominal: np.ndarray             # (3,) λ ХХ без демага при той же T [Вб]
    lam_after: np.ndarray               # (3,) λ ХХ после (необратимого) демага [Вб]
    flux_linkage_drop_frac: float       # относит. падение амплитуды λ = падение ЭДС/момента
    load_converged: bool


def _fixed_magnet_source(nc: int, idx: np.ndarray, axes: np.ndarray,
                         nu_rec: float, br_par: np.ndarray) -> np.ndarray:
    """Фиксированный источник магнита ν·B_r·e по ячейкам (линейный recoil, без колена)."""
    src = np.zeros((nc, 2), dtype=float)
    src[idx] = (nu_rec * np.asarray(br_par, dtype=float))[:, None] * axes[idx]
    return src


def evaluate_demag_impact(
    geometry: MachineGeometry,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
    layout: WindingLayout,
    *,
    i_peak: float,
    gamma_elec: float,
    turns_per_slot: float,
    T: float = 20.0,
    relaxation: float = 0.1,
    max_iter: int = 300,
) -> DemagImpact:
    """
    Полный машинный итог демага (3 решения на общей сборке):
      1) worst-case нагрузка (ток+T) → сошедшееся поле → рабочая точка H_par каждой ячейки;
      2) заморозить необратимое B_r_eff = effective_Br(H_par_нагрузки, T) (интерцепт линии
         возврата) — состояние магнита ПОСЛЕ снятия нагрузки (необратимо);
      3) сравнить λ холостого хода при ФИКСИРОВАННОМ источнике: номинал B_r(T) vs
         замороженный B_r_eff. Падение амплитуды = относит. падение ЭДС- и моментной постоянной.
    Обе λ при ОДНОЙ T и на линии возврата ⇒ изолирует НЕОБРАТИМУЮ потерю (обратимое T-падение
    Br входит в обе одинаково и сокращается). H_par берётся из СОШЕДШЕГОСЯ поля, а не из
    latching по итерациям Picard (тот ловит численные переходные выбросы → переоценка демага).
    """
    space = LagrangeP1Space2D(geometry.mesh)
    nc = geometry.mesh.n_cells
    nu_of_B, nu_init, magnet_mask, _ = machine_reluctivity(geometry, magnet, steel)
    axes = geometry.magnet_easy_axis
    idx = np.where(magnet_mask)[0]
    nu_rec = 1.0 / magnet.mu_rec

    def _noload_fixed(br_par):
        src = _fixed_magnet_source(nc, idx, axes, nu_rec, br_par)
        return solve_nonlinear_2d_picard(
            space, nu_of_B=nu_of_B, nu_init=nu_init, magnetization=src,
            relaxation=relaxation, max_iter=max_iter,
        )

    # 1) worst-case нагрузка (с коленом, БЕЗ latching — берём сошедшееся поле).
    pol = MagnetDemagPolicy(magnet, magnet_mask, T=T, n_cells=nc, axis=axes,
                            relaxation=relaxation)
    jz = MU0 * winding_current_density(
        geometry, layout, i_peak=i_peak, gamma_elec=gamma_elec, turns_per_slot=turns_per_slot
    )
    em_load = solve_nonlinear_2d_picard(
        space, nu_of_B=nu_of_B, nu_init=nu_init, j_cells=jz, magnetization=pol,
        relaxation=relaxation, max_iter=max_iter,
    )
    risk = compute_demag_risk_map(magnet, em_load, magnet_mask, T=T, axis=axes)

    # 2) заморозить B_r_eff по сошедшемуся нагруженному H (линия возврата).
    br_eff_frozen = np.asarray(magnet.effective_Br(risk.H_par, T), dtype=float)
    br_nom_par = np.full(idx.size, float(magnet.Br(T)))

    # 3) ХХ с фиксированным источником: номинал vs замороженный.
    lam_nom = phase_flux_linkage(geometry, layout, _noload_fixed(br_nom_par).a,
                                 turns_per_slot=turns_per_slot)
    lam_after = phase_flux_linkage(geometry, layout, _noload_fixed(br_eff_frozen).a,
                                   turns_per_slot=turns_per_slot)

    amp_nom = _clarke_amplitude(lam_nom)
    amp_after = _clarke_amplitude(lam_after)
    drop = 1.0 - amp_after / amp_nom if amp_nom > 0.0 else 0.0
    return DemagImpact(
        risk=risk,
        aggregate=magnet_loss_aggregate(geometry, risk),
        lam_nominal=lam_nom,
        lam_after=lam_after,
        flux_linkage_drop_frac=float(drop),
        load_converged=em_load.converged,
    )
