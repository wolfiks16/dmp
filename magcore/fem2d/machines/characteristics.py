from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.fem2d.losses import copper_resistivity
from magcore.fem2d.machines.bridge import pmsm_to_problem
from magcore.fem2d.machines.excitation import slot_areas, winding_current_density
from magcore.fem2d.machines.loss import phase_flux_linkage
from magcore.fem2d.machines.postproc import (
    back_emf_constant,
    flux_linkage_amplitude,
    torque_constant,
)
from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry, Region
from magcore.fem2d.machines.scenario import MachineScenario
from magcore.fem2d.machines.static_solver import machine_reluctivity
from magcore.fem2d.model.postproc import torque_arkkio
from magcore.fem2d.model.problem import Solution2D
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D

# ХАРАКТЕРИСТИКИ МАШИНЫ — то, что отвечает на инженерный вопрос «сколько это стоит», в
# отличие от карты риска, отвечающей «где именно». Считаются для ЛЮБОГО состояния магнита:
# целого или уже повреждённого (доля сохранённой ремнантности r по ячейкам), поэтому один
# и тот же код даёт снимок «до» и «после» перегрузки.
#
# Что здесь есть: момент при заданном токе, моментная постоянная K_t=(3/2)p·λ_m и ЭДС-
# постоянная K_e=p·λ_m (это РАЗНЫЕ величины, отличаются в 3/2 раза), амплитуда
# потокосцепления ПМ λ_m, и — как прямая мера ЭФФЕКТИВНОСТИ МАГНИТОВ — момент на килограмм
# магнита (важно при сравнении NdFeB и SmCo: у SmCo ниже B_r, и вопрос, во что обходится
# его температурная стойкость по массе редкозёмов).
#
# ⚠ ГРАНИЦА: вращение ротора не реализовано, поэтому величины берутся при ОДНОМ его
# положении. Для целой симметричной машины это практически среднее значение, но при
# НЕСИММЕТРИЧНОМ повреждении (реакция якоря бьёт по конкретным полюсам) появляются
# пульсации, и разница «мгновенное против среднего» становится содержательной. Поэтому
# рядом всегда возвращается оценка первого порядка через инвариантную к положению ротора
# первую гармонику ремнантности: расхождение этих двух чисел И ЕСТЬ мера несимметрии.


MAGNET_DENSITY_DEFAULT = 7500.0     # NdFeB [кг/м³]; Sm2Co17 ≈ 8400 — задавать явно


@dataclass(frozen=True, slots=True)
class MachineCharacteristics:
    """Снимок характеристик машины при заданном токе, температуре и состоянии магнита."""

    torque: float                 # электромагнитный момент [Н·м] (Арккио по зазору)
    torque_constant: float        # K_t = (3/2)·p·λ_m [Н·м/А] — момент на амплитуду тока q
    emf_constant: float           # K_e = p·λ_m [В·с/рад] — пик фазной ЭДС на рад/с
    flux_linkage: float           # амплитуда потокосцепления ПМ λ_m [Вб] (холостой ход)
    magnet_mass: float            # масса магнитов [кг]
    i_peak: float                 # ток, при котором снят момент [А]
    T: float                      # температура магнита [°C]
    converged: bool

    @property
    def torque_per_magnet_mass(self) -> float:
        """Момент на килограмм магнита [Н·м/кг] — прямая мера эффективности магнитов."""
        return float(self.torque / self.magnet_mass) if self.magnet_mass > 0.0 else 0.0


def magnet_mass(
    geometry: MachineGeometry, *, density: float = MAGNET_DENSITY_DEFAULT
) -> float:
    """Масса магнитов [кг] = площадь сечения × осевая длина × плотность."""
    mesh = geometry.mesh
    idx = np.where(geometry.mask(Region.MAGNET))[0]
    area = float(sum(mesh.cell_area(int(c)) for c in idx))
    return area * geometry.params.axial_length * float(density)


def _solve_with_retention(
    scenario: MachineScenario, *, T: float, retention: np.ndarray | None,
    j_cells: np.ndarray | None, relaxation: float, max_iter: int, tol: float,
):
    """
    Магнитостатика при ЗАМОРОЖЕННОМ состоянии магнита: источник ν·B_r с B_r = r·B_r(T).

    Обратная связь по полю здесь намеренно ОТКЛЮЧЕНА (источник статический): состояние уже
    определено историей нагрева, магнит живёт на линии возврата, и повторный запуск
    demag-политики только внёс бы новые потери от снимаемого режима.
    """
    geometry = scenario.geometry
    space = LagrangeP1Space2D(geometry.mesh)
    nu_of_B, nu_init, magnet_mask, _ = machine_reluctivity(geometry, scenario.magnet, scenario.steel)
    idx = np.where(magnet_mask)[0]

    br = np.full(idx.size, float(scenario.magnet.Br(T)), dtype=float)
    if retention is not None:
        r = np.asarray(retention, dtype=float).reshape(-1)
        if r.shape != idx.shape:
            raise ValueError("retention должен быть по ячейкам магнита.")
        br = br * r

    nu_br = np.zeros((geometry.mesh.n_cells, 2), dtype=float)
    nu_br[idx] = (br / scenario.magnet.mu_rec)[:, None] * geometry.magnet_easy_axis[idx]

    return solve_nonlinear_2d_picard(
        space, nu_of_B=nu_of_B, nu_init=nu_init,
        j_cells=(None if j_cells is None else MU0 * j_cells),
        magnetization=nu_br, relaxation=relaxation, max_iter=max_iter, tol=tol,
    )


def machine_characteristics(
    scenario: MachineScenario,
    *,
    i_peak: float,
    turns_per_slot: float,
    gamma_elec: float = 0.0,
    T: float = 20.0,
    retention: np.ndarray | None = None,
    magnet_density: float = MAGNET_DENSITY_DEFAULT,
    relaxation: float = 0.1,
    max_iter: int = 300,
    tol: float = 1.0e-6,
) -> MachineCharacteristics:
    """
    Снять характеристики машины при заданном режиме и состоянии магнита.

    Требуется ДВА решения: под нагрузкой (для момента) и на холостом ходу (для λ_m и K_t —
    моментная постоянная определяется потокосцеплением ПМ, а не моментом, делённым на ток:
    при насыщении и реакции якоря второе от тока зависит и постоянной не является).

    `retention=None` — целый магнит; иначе доля сохранённой ремнантности по его ячейкам
    (например `MachineThermalDemagResult.retention`).
    """
    geometry = scenario.geometry
    p = geometry.params

    jz = winding_current_density(
        geometry, scenario.layout, i_peak=i_peak, gamma_elec=gamma_elec,
        turns_per_slot=turns_per_slot,
    )
    kw = dict(T=T, retention=retention, relaxation=relaxation, max_iter=max_iter, tol=tol)
    em_load = _solve_with_retention(scenario, j_cells=jz, **kw)
    em_open = _solve_with_retention(scenario, j_cells=None, **kw)

    problem = pmsm_to_problem(geometry, scenario.magnet, scenario.steel, T=T, j_cells=jz)
    torque = torque_arkkio(
        Solution2D(problem=problem, field=em_load, risk=None),
        p.R_s_out, p.R_mag_in, axial_length=p.axial_length,
    )

    lam = phase_flux_linkage(
        geometry, scenario.layout, em_open.a, turns_per_slot=turns_per_slot
    )
    lam_m = flux_linkage_amplitude(lam)

    return MachineCharacteristics(
        torque=float(torque),
        torque_constant=torque_constant(geometry, lam),
        emf_constant=back_emf_constant(geometry, lam),
        flux_linkage=float(lam_m),
        magnet_mass=magnet_mass(geometry, density=magnet_density),
        i_peak=float(i_peak), T=float(T),
        converged=bool(em_load.converged and em_open.converged),
    )


@dataclass(frozen=True, slots=True)
class CharacteristicsComparison:
    """Что перегрузка стоила машине: характеристики до и после, и относительные потери."""

    before: MachineCharacteristics
    after: MachineCharacteristics
    fundamental_ratio: float      # инвариантная к положению ротора оценка сохранности

    @staticmethod
    def _drop(a: float, b: float) -> float:
        return float((a - b) / a) if a != 0.0 else 0.0

    @property
    def torque_drop(self) -> float:
        return self._drop(self.before.torque, self.after.torque)

    @property
    def torque_constant_drop(self) -> float:
        return self._drop(self.before.torque_constant, self.after.torque_constant)

    @property
    def flux_linkage_drop(self) -> float:
        return self._drop(self.before.flux_linkage, self.after.flux_linkage)

    @property
    def asymmetry_indicator(self) -> float:
        """
        Насколько «мгновенная» просадка K_t расходится с инвариантной оценкой по первой
        гармонике. Малое значение — повреждение почти симметрично, снятые при одном положении
        ротора числа представительны. Большое — повреждение сосредоточено на части полюсов,
        и корректные средние значения требуют прогонки по положениям ротора (не реализовано).
        """
        return abs(self.torque_constant_drop - (1.0 - self.fundamental_ratio))


# --- P-B6: режим напряжения (деградация характеристик при нагреве) ---

def tooth_coil_end_length(geometry: MachineGeometry) -> float:
    """
    Длина ЛОБОВЫХ соединений на ОДИН виток зубцовой катушки [м] (оба торца вместе).

    Конструкция всех серийных БПЛА-аутраннеров с дробной обмоткой (36N40P, 12N14P …):
    каждый зуб намотан своей катушкой, сторона катушки занимает ПОЛОВИНУ паза у своего зуба.
    Центры двух сторон одной катушки разнесены на c = R_ср·(θ_зуба + θ_паза/2) по дуге
    среднего радиуса зубцовой зоны; лобовая часть огибает торец зуба полуокружностью
    диаметра c ⇒ π·c/2 на торец, π·c на виток.

    ⚠ Оценка геометрическая (реальная лобовая часть — скруглённый прямоугольник, с вылетом
    на изоляцию торца). Для короткого пакета она СУЩЕСТВЕННА: у IM-8008 (пакет 8 мм) лобовые
    части почти удваивают сопротивление — ровно поэтому их нельзя опускать у «блинов».
    """
    p = geometry.params
    slot_pitch = 2.0 * np.pi / p.n_slots
    tooth_ang = p.tooth_width_frac * slot_pitch
    slot_ang = slot_pitch - tooth_ang
    r_mid = 0.5 * (p.R_sy + p.R_s_out)
    c = r_mid * (tooth_ang + 0.5 * slot_ang)
    return float(np.pi * c)


def phase_resistance(
    geometry: MachineGeometry, *, turns_per_slot: float, slot_fill: float, T: float,
    end_length_per_turn: float = 0.0, ac_factor: float = 1.0,
) -> float:
    """
    Сопротивление фазы R(T) [Ом], согласованное с моделью потерь меди:
    R_фазы = (1/3)·Σ_пазов ρ_cu(T)·(F_R·L + l_лоб/2)·N²/(k_зап·A_паз). Растёт с T через ρ_cu(T).

    Вывод: суммарные потери меди P = Σ ρ(T)·J²·V = I_скз²·Σ ρ(T)·N²·L/(k·A_паз), а
    P = 3·I_скз²·R_фазы ⇒ R_фазы = (1/3)·Σ ρ(T)·N²·L/(k·A_паз).

    `end_length_per_turn` — лобовые части на ВИТОК, оба торца [м] (см. `tooth_coil_end_length`);
        на одну сторону катушки (проводник паза) приходится половина. 0 — прежнее поведение
        (только активная длина: занижает R; у короткого пакета — почти вдвое).
    `ac_factor` — F_R = R_ac/R_dc для части В ПАЗУ (см. `losses.dowell_ac_factor`); лобовые
        части в воздухе, к ним он не применяется. 1 — постоянный ток.
    """
    if not (0.0 < slot_fill <= 1.0):
        raise ValueError("slot_fill in (0, 1].")
    if end_length_per_turn < 0.0:
        raise ValueError("end_length_per_turn must be non-negative.")
    if ac_factor < 1.0:
        raise ValueError("ac_factor must be >= 1 (переменный ток сопротивление не уменьшает).")
    A_slot = np.asarray(slot_areas(geometry), dtype=float)
    L = geometry.params.axial_length
    length = float(ac_factor) * L + 0.5 * float(end_length_per_turn)
    rho = float(copper_resistivity(T))
    R_slots = rho * length * float(turns_per_slot) ** 2 / (float(slot_fill) * A_slot)
    return float(R_slots.sum() / 3.0)


def voltage_limited_operating_point(
    *, voltage: float, omega_mech: float, emf_constant: float, resistance: float,
    core_loss_w: float = 0.0,
) -> dict:
    """
    Рабочая точка под ОГРАНИЧЕНИЕМ НАПРЯЖЕНИЯ (лумпед DC-эквивалент BLDC — для иллюстрации
    ДЕГРАДАЦИИ характеристик, не точного КПД): противо-ЭДС E=K_e·ω, ток I=(U−E)/R (зажат ≥0),
    момент = K_e·I (K_t≡K_e в лумпед-модели), P_мех=E·I. Реактивность пренебрежена (она
    ~T-независима и не влияет на ДЕЛЬТУ деградации). Нагрев бьёт двояко: R(T)↑ и K_e(T,демаг)↓.

    Возвращает dict(current, back_emf, torque, p_mech, p_cu, efficiency).
    """
    if resistance <= 0.0:
        raise ValueError("resistance must be positive.")
    E = float(emf_constant) * float(omega_mech)
    I = max(float(voltage) - E, 0.0) / float(resistance)
    torque = float(emf_constant) * I
    p_mech = E * I
    p_cu = I * I * float(resistance)
    denom = p_mech + p_cu + float(core_loss_w)
    eff = float(p_mech / denom) if denom > 0.0 else 0.0
    return dict(current=I, back_emf=E, torque=torque, p_mech=p_mech, p_cu=p_cu, efficiency=eff)
