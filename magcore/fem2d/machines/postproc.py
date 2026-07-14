from __future__ import annotations

import numpy as np

from magcore.constants import MU0
from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry

# P6: ЧИСТО СТАТИЧЕСКИЙ пост-процессинг из решённого поля — момент и ЭДС/потокосцепление.
# (Потери/нагрев сюда НЕ входят: нагрев транзиентен ⇒ динамический модуль S2. Решение Sergey.)
#
# Момент по Арккио: усреднённый по зазорному кольцу тензор Максвелла,
#   T = L/(μ₀·(r_o−r_i)) · ∫∫_gap r·B_r·B_θ dA,
# где B_r, B_θ — радиальная/тангенц. компоненты в зазоре r∈[R_s_out, R_mag_in]. Устойчивее
# линейного интеграла на одной окружности (усреднение по толщине зазора гасит сеточный шум).


def airgap_cell_mask(geometry: MachineGeometry) -> np.ndarray:
    """Ячейки ЗАЗОРА (кольцо между кончиками зубьев и магнитами) по радиусу центроида."""
    p = geometry.params
    mesh = geometry.mesh
    cen = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)])
    r = np.hypot(cen[:, 0], cen[:, 1])
    return (r >= p.R_s_out) & (r <= p.R_mag_in)


def airgap_torque_arkkio(
    geometry: MachineGeometry, B_cells: np.ndarray, *, axial_length: float | None = None
) -> float:
    """
    Электромагнитный момент [Н·м] по методу Арккио из поля `B_cells` (Тл).
    T = L/(μ₀(r_o−r_i))·Σ_{c∈gap} r_c·B_r,c·B_θ,c·area_c. Знак = направление момента на ротор.
    """
    p = geometry.params
    L = p.axial_length if axial_length is None else float(axial_length)
    mesh = geometry.mesh
    r_i, r_o = p.R_s_out, p.R_mag_in

    cen = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)])
    r = np.hypot(cen[:, 0], cen[:, 1])
    gap = (r >= r_i) & (r <= r_o)
    if not np.any(gap):
        return 0.0
    idx = np.where(gap)[0]
    rc = r[idx]
    rhat = cen[idx] / rc[:, None]
    that = np.stack([-cen[idx, 1], cen[idx, 0]], axis=1) / rc[:, None]  # θ̂ = (−y, x)/r
    Bx, By = B_cells[idx, 0], B_cells[idx, 1]
    Br = Bx * rhat[:, 0] + By * rhat[:, 1]
    Bth = Bx * that[:, 0] + By * that[:, 1]
    areas = np.array([mesh.cell_area(int(c)) for c in idx])
    integral = float(np.sum(rc * Br * Bth * areas))
    return L / (MU0 * (r_o - r_i)) * integral


def flux_linkage_amplitude(lam3: np.ndarray) -> float:
    """Амплитуда потокосцепления ПМ из фазных λ (Кларк, инвариант амплитуды) [Вб]."""
    lam = np.asarray(lam3, dtype=float)
    alpha = lam[0] - 0.5 * lam[1] - 0.5 * lam[2]
    beta = (np.sqrt(3.0) / 2.0) * (lam[1] - lam[2])
    return float(np.hypot(alpha, beta))


def back_emf_constant(geometry: MachineGeometry, lam3: np.ndarray) -> float:
    """
    ЭДС-постоянная K_e = p·λ_m [В·с/рад] (= моментной постоянной K_t в СИ). λ_m — амплитуда
    потокосцепления ПМ холостого хода (из `phase_flux_linkage`), p = число пар полюсов.
    Пик фазной ЭДС = K_e·ω_mech. Падение λ_m из-за демага (P5) = падение K_e и K_t.
    """
    p = geometry.params.n_poles // 2
    return float(p * flux_linkage_amplitude(lam3))
