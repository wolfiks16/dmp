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
    """
    Амплитуда потокосцепления ПМ из фазных λ (Кларк, ИНВАРИАНТ АМПЛИТУДЫ) [Вб].

    Множитель 2/3 обязателен: без него на сбалансированном наборе с амплитудой λ функция
    возвращает 1.5·λ. Проверяется тестом на наборе с ЗАВЕДОМО известной амплитудой.
    """
    lam = np.asarray(lam3, dtype=float)
    alpha = (2.0 / 3.0) * (lam[0] - 0.5 * lam[1] - 0.5 * lam[2])
    beta = (2.0 / 3.0) * (np.sqrt(3.0) / 2.0) * (lam[1] - lam[2])
    return float(np.hypot(alpha, beta))


def back_emf_constant(geometry: MachineGeometry, lam3: np.ndarray) -> float:
    """
    ЭДС-постоянная K_e = p·λ_m [В·с/рад]: пик ФАЗНОЙ ЭДС = K_e·ω_mech. λ_m — амплитуда
    потокосцепления ПМ холостого хода, p = число пар полюсов.

    ⚠ K_e ≠ K_t: моментная постоянная больше в 3/2 раза (три фазы), см. `torque_constant`.
    Прежняя версия утверждала равенство и вдобавок опиралась на амплитуду Кларка без
    множителя 2/3 — то есть возвращала (3/2)·p·λ_m: верное K_t под именем K_e.
    """
    p = geometry.params.n_poles // 2
    return float(p * flux_linkage_amplitude(lam3))


def torque_constant(geometry: MachineGeometry, lam3: np.ndarray) -> float:
    """
    Моментная постоянная K_t = (3/2)·p·λ_m [Н·м/А]: момент на АМПЛИТУДУ тока по оси q.
    Множитель 3/2 — вклад трёх фаз. Сверено с моментом Арккио из поля (см. тест):
    при малом токе (до насыщения) M_max/I_peak сходится к этой величине.
    """
    p = geometry.params.n_poles // 2
    return float(1.5 * p * flux_linkage_amplitude(lam3))
