from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.fem2d.machines.bridge import pmsm_to_problem
from magcore.fem2d.machines.excitation import winding_current_density
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
