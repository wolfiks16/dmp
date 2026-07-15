from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.excitation import winding_current_density
from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry, Region
from magcore.fem2d.machines.winding import WindingLayout
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.hybrid.magnet_demag import (
    DemagRiskMap,
    MagnetDemagPolicy,
    compute_demag_risk_map,
)

# P4: полный СВЯЗАННЫЙ статический решатель в реальной геометрии машины — сводит воедино
# (1) магнит с необратимым коленом (MagnetDemagPolicy, поячеечная радиальная ось),
# (2) нелинейную сталь ярма/зубьев ν(|B|), (3) ток обмотки из P3, на общем Picard.
# Сценарии: S1 (T=20°, поле только от обмоток+магнит) и S2 (T задана).
#
# КОНВЕНЦИЯ ЕДИНИЦ (относительная, как во всём fem2d/demag): ν = 1/μ_r (воздух=1,
# магнит=1/μ_rec, сталь=μ₀·ν_chord(|B|)); B [Тл]; H_solver = μ₀·H_физ; источник тока в
# правой части = μ₀·J (умножаем здесь). Тогда −div(ν∇A_z)=μ₀J+curl(ν·B_r) физически верна,
# а мост H_физ = H_solver/μ₀ (в demag-политике) самосогласован.

_STEEL_REGIONS = (int(Region.STATOR_YOKE), int(Region.TOOTH), int(Region.ROTOR_YOKE))


@dataclass(frozen=True, slots=True)
class MachineStaticResult:
    a: np.ndarray               # (ndofs,) узловой A_z [Тл·м]
    B_cells: np.ndarray         # (n_cells, 2) [Тл]
    H_cells: np.ndarray         # (n_cells, 2) H_solver = μ₀·H_физ [Тл]
    nu_cells: np.ndarray        # (n_cells,) относительная ν финальной сборки
    converged: bool
    n_iterations: int
    T: float                    # температура сценария [°C]
    gamma_elec: float | None    # эл. угол тока (None, если тока нет)
    risk: DemagRiskMap          # карта риска размагничивания магнита


def machine_reluctivity(
    geometry: MachineGeometry,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
):
    """
    Поячеечный ν_of_B (ОТНОСИТЕЛЬНЫЙ 1/μ_r) + начальное ν + маски магнита/стали.
    Воздух/зазор/паз → 1; магнит → 1/μ_rec (recoil, T-независим); сталь → μ₀·ν_chord(|B|).
    Возвращает (nu_of_B, nu_init, magnet_mask, steel_mask).
    """
    region = geometry.region
    nc = geometry.mesh.n_cells
    magnet_mask = region == int(Region.MAGNET)
    steel_mask = np.isin(region, _STEEL_REGIONS)
    steel_idx = np.where(steel_mask)[0]
    nu_mag = 1.0 / magnet.mu_rec

    def nu_of_B(B_cells: np.ndarray) -> np.ndarray:
        nu = np.ones(nc, dtype=float)              # воздух/зазор/паз
        nu[magnet_mask] = nu_mag                   # магнит: recoil
        if steel_idx.size:
            Bmag = np.hypot(B_cells[steel_idx, 0], B_cells[steel_idx, 1])
            nu[steel_idx] = MU0 * np.array([steel.nu_chord(float(b)) for b in Bmag])
        return nu

    nu_init = nu_of_B(np.zeros((nc, 2), dtype=float))
    return nu_of_B, nu_init, magnet_mask, steel_mask


def solve_machine_static(
    geometry: MachineGeometry,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
    *,
    T: float = 20.0,
    layout: WindingLayout | None = None,
    i_peak: float = 0.0,
    gamma_elec: float = 0.0,
    turns_per_slot: float = 0.0,
    track_worst_point: bool = False,
    relaxation: float = 0.1,
    max_iter: int = 150,
    tol: float = 1.0e-6,
) -> MachineStaticResult:
    """
    Статический связанный solve на сетке машины. Магнит(демаг)+сталь(|B|)+ток обмотки →
    общий Picard. S1: T=20, ток по желанию; S2: T задана. Ток включается, когда заданы
    layout+i_peak+turns_per_slot (иначе только магнит). ГУ по умолчанию: A_z=0 на границе
    сетки (внутренняя расточка + внешняя поверхность ротора) — поток заперт в ярмах.

    ⚠ Сходимость: хордовый Picard по сильно насыщающейся стали (ν_rel скачет от ~2e-4 до 1)
    склонен к предельному циклу при слабой релаксации; relaxation≈0.1 даёт монотонную
    сходимость (≈85–90 итераций на грубой сетке; магнит+ток+нагрев). Тоньше сетка/сильнее
    привод — меньше релаксация (или Ньютон через ν_d — задел на будущее).
    """
    space = LagrangeP1Space2D(geometry.mesh)
    nc = geometry.mesh.n_cells
    nu_of_B, nu_init, magnet_mask, _ = machine_reluctivity(geometry, magnet, steel)

    j_cells = None
    have_current = layout is not None and i_peak != 0.0 and turns_per_slot != 0.0
    if have_current:
        jz = winding_current_density(
            geometry, layout, i_peak=i_peak, gamma_elec=gamma_elec,
            turns_per_slot=turns_per_slot,
        )
        j_cells = MU0 * jz                         # относительная конвенция: RHS = μ₀·J

    axes = geometry.magnet_easy_axis               # (nc,2) радиальная ось·полярность
    policy = MagnetDemagPolicy(
        magnet, magnet_mask, T=T, n_cells=nc, axis=axes,
        relaxation=relaxation, track_worst_point=track_worst_point,
    )

    em = solve_nonlinear_2d_picard(
        space, nu_of_B=nu_of_B, nu_init=nu_init, j_cells=j_cells,
        magnetization=policy, relaxation=relaxation, max_iter=max_iter, tol=tol,
    )
    risk = compute_demag_risk_map(magnet, em, magnet_mask, T=T, axis=axes)
    return MachineStaticResult(
        a=em.a, B_cells=em.B_cells, H_cells=em.H_cells, nu_cells=em.nu_cells,
        converged=em.converged, n_iterations=em.n_iterations,
        T=float(T), gamma_elec=(float(gamma_elec) if have_current else None), risk=risk,
    )


def worst_case_gamma_sweep(
    geometry: MachineGeometry,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
    layout: WindingLayout,
    *,
    i_peak: float,
    turns_per_slot: float,
    T: float = 20.0,
    n_angles: int = 12,
    relaxation: float = 0.1,
    max_iter: int = 150,
) -> tuple[MachineStaticResult, np.ndarray, np.ndarray]:
    """
    Найти ИСТИННЫЙ worst-case демага перебором электрического угла тока γ ∈ [0,2π):
    для каждого γ решить и взять минимум маржи к колену по магниту; вернуть худшее
    решение + (углы, маржи). Нужно решённое поле — потому свип, а не аналитика (P3→P4).
    """
    gammas = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)
    margins = np.empty(gammas.shape, dtype=float)
    results: list[MachineStaticResult] = []
    for i, g in enumerate(gammas):
        r = solve_machine_static(
            geometry, magnet, steel, T=T, layout=layout, i_peak=i_peak,
            gamma_elec=float(g), turns_per_slot=turns_per_slot,
            relaxation=relaxation, max_iter=max_iter,
        )
        results.append(r)
        margins[i] = r.risk.worst_margin
    worst = int(np.argmin(margins))
    return results[worst], gammas, margins
