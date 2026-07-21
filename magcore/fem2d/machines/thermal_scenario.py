from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from magcore.constants import MU0
from magcore.fem2d.coupled_transient import (
    CoupledTransientResult,
    solve_coupled_magneto_thermal_transient,
)
from magcore.fem2d.machines.excitation import slot_areas, winding_current_density
from magcore.fem2d.machines.pmsm_outrunner import REGION_NAMES, MachineGeometry, Region
from magcore.fem2d.machines.scenario import MachineScenario
from magcore.fem2d.machines.static_solver import machine_reluctivity
from magcore.fem2d.spaces import LagrangeP1Space2D

# S3, инкремент 4: сценарий «ТЕПЛОВАЯ СТОЙКОСТЬ МАГНИТА» на РЕАЛЬНОЙ машине.
# Переносит верифицированную на сегменте связку (coupled_transient) на сечение outrunner PMSM
# и доводит результат до ИНЖЕНЕРНОГО итога: на сколько просела способность машины создавать
# момент из-за НЕОБРАТИМОЙ потери ремнантности, накопленной вдоль траектории нагрева.
#
# ДВА РАЗНЫХ ТОКА (принципиально, легко ошибиться):
#   * магнитостатика — МГНОВЕННЫЙ ток при заданном угле γ (худший случай реакции якоря;
#     именно он определяет, куда уедет рабочая точка магнита);
#   * нагрев — СРЕДНЕКВАДРАТИЧНЫЙ за электрический период (тепловая постоянная времени
#     машины — десятки секунд, период — миллисекунды ⇒ греет среднее, не мгновенное),
#     и дополнительно поправленный на коэффициент заполнения паза: ток размазан по всей
#     площади паза, а джоулево тепло выделяет только медь ⇒ q = ρ(T)·J_паз²/k_зап.
#     Забыть k_зап — занизить потери примерно вдвое.
#
# ИНЖЕНЕРНЫЙ ИТОГ считается по ПЕРВОЙ ГАРМОНИКЕ распределения ремнантности вокруг ротора,
# а не по фазному потокосцеплению при одном положении ротора. Причина: реакция якоря бьёт
# по КОНКРЕТНЫМ полюсам, повреждение выходит НЕСИММЕТРИЧНЫМ, и Кларк-амплитуда несимметричного
# набора λ перестаёт измерять K_e — на практике давала даже ОТРИЦАТЕЛЬНОЕ падение
# («повреждённый магнит сильнее»), что физически невозможно. Корректная амплитуда потребовала
# бы прогонки по положениям ротора (вращение не реализовано).
# Первая гармоника же берётся в системе координат РОТОРА и от его положения не зависит:
#     ratio = |Σ A_c·B_r_eff,c·e^{i·p·θ_c}| / |Σ A_c·B_r,c·e^{i·p·θ_c}|,
# где полярность полюса входит через знак ремнантности. В линейной магнитной цепи K_e ∝ этой
# гармонике (обмоточный коэффициент не меняется), поэтому ratio — оценка сохранившейся
# моментной/ЭДС-постоянной ПЕРВОГО ПОРЯДКА; насыщение стали даёт поправку второго порядка.


@dataclass(frozen=True)
class MachineThermalProperties:
    """Теплопроводность k [Вт/(м·К)] и объёмная теплоёмкость c=ρ·c_p [Дж/(м³·К)] по регионам."""

    k_by_region: Mapping[str, float]
    c_by_region: Mapping[str, float]

    @staticmethod
    def representative() -> "MachineThermalProperties":
        """
        Представительный набор для электрической машины. ⚠ ориентировочные значения —
        под конкретное изделие уточнять (особенно паз: это ГОМОГЕНИЗИРОВАННАЯ смесь
        медь+изоляция+пропитка, чья эффективная поперечная теплопроводность на порядок ниже
        чистой меди, и зазор: там реально работает конвекция, а не теплопроводность воздуха).
        """
        return MachineThermalProperties(
            k_by_region={
                "air_gap": 0.5, "stator_yoke": 25.0, "tooth": 25.0,
                "slot": 1.0, "magnet": 9.0, "rotor_yoke": 25.0,
            },
            c_by_region={
                "air_gap": 1.0e4, "stator_yoke": 3.5e6, "tooth": 3.5e6,
                "slot": 2.5e6, "magnet": 3.0e6, "rotor_yoke": 3.5e6,
            },
        )

    def cell_fields(self, geometry: MachineGeometry) -> tuple[np.ndarray, np.ndarray]:
        """Разложить свойства по ячейкам сетки: (k_cells, c_cells)."""
        nc = geometry.mesh.n_cells
        k = np.zeros(nc, dtype=float)
        c = np.zeros(nc, dtype=float)
        for code, name in REGION_NAMES.items():
            sel = geometry.region == code
            if not sel.any():
                continue
            if name not in self.k_by_region or name not in self.c_by_region:
                raise ValueError(f"нет тепловых свойств для региона {name!r}.")
            k[sel] = float(self.k_by_region[name])
            c[sel] = float(self.c_by_region[name])
        if not np.all(k > 0.0) or not np.all(c > 0.0):
            raise ValueError("k и c должны быть положительны во всех ячейках.")
        return k, c


@dataclass(frozen=True, slots=True)
class MachineThermalDemagResult:
    """Итог сценария: траектория связки + инженерные последствия для машины."""

    transient: CoupledTransientResult
    retention: np.ndarray         # (n_magnet_cells,) итоговая доля сохранённой ремнантности
    fundamental_ratio: float      # доля сохранившейся 1-й гармоники ремнантности ротора
    loss_power_initial: float     # потери в меди в начале [Вт/м] (на единицу осевой длины)
    loss_power_final: float       # потери в конце (выросли из-за ρ(T))

    @property
    def torque_constant_drop(self) -> float:
        """
        ОЦЕНКА относительного падения моментной/ЭДС-постоянной (0 — магнит цел).
        Первого порядка: K_e ∝ первой гармонике ремнантности при неизменной обмотке
        (см. шапку модуля); насыщение стали даёт поправку второго порядка.
        """
        return float(1.0 - self.fundamental_ratio)

    @property
    def T_magnet_max(self) -> float:
        return float(np.nanmax(self.transient.T_magnet))

    @property
    def survived(self) -> bool:
        """Расчёт дошёл до конца горизонта (не сорвался в разгон/каскад)."""
        return not (self.transient.runaway or self.transient.magnet_cascade)


def _slot_rms_loss_current(
    geometry: MachineGeometry, *, i_peak: float, turns_per_slot: float, slot_fill: float
) -> np.ndarray:
    """
    Плотность тока для ПОТЕРЬ по ячейкам: J = N·I_rms/A_паз / √k_зап (см. шапку модуля).

    Для симметричной синусоидальной трёхфазной системы среднеквадратичный ток КАЖДОЙ фазы
    равен I_peak/√2 независимо от угла γ, поэтому величина от γ не зависит (в отличие от
    мгновенной) — что и требуется тепловой задаче.
    """
    if not (0.0 < slot_fill <= 1.0):
        raise ValueError("slot_fill in (0, 1].")
    sid = geometry.slot_id
    A = slot_areas(geometry)
    j = np.zeros(geometry.mesh.n_cells, dtype=float)
    in_slot = sid >= 0
    i_rms = float(i_peak) / math.sqrt(2.0)
    j[in_slot] = float(turns_per_slot) * i_rms / A[sid[in_slot]] / math.sqrt(float(slot_fill))
    return j


def magnet_fundamental_ratio(
    geometry: MachineGeometry, retention: np.ndarray, magnet_mask: np.ndarray
) -> float:
    """
    Доля сохранившейся ПЕРВОЙ ГАРМОНИКИ ремнантности ротора (см. шапку модуля).
    1.0 — магнит цел; 0.9 — первая гармоника просела на 10 %.

    Величина инвариантна к положению ротора и к обратимому температурному падению B_r
    (оно входит множителем в числитель и знаменатель и сокращается) — остаётся ровно
    накопленная НЕОБРАТИМАЯ потеря, распределённая по полюсам.
    """
    idx = np.where(np.asarray(magnet_mask, dtype=bool))[0]
    r = np.asarray(retention, dtype=float).reshape(-1)
    if r.shape != idx.shape:
        raise ValueError("retention должен быть по ячейкам магнита.")
    mesh = geometry.mesh
    p = geometry.params.n_poles // 2

    cent = np.array([mesh.cell_vertices(int(c)).mean(axis=0) for c in idx], dtype=float)
    theta = np.arctan2(cent[:, 1], cent[:, 0])
    area = np.array([mesh.cell_area(int(c)) for c in idx], dtype=float)

    # Полярность полюса: знак проекции лёгкой оси на радиальное направление.
    radial = cent / np.linalg.norm(cent, axis=1)[:, None]
    pol = np.sign(np.einsum("ij,ij->i", geometry.magnet_easy_axis[idx], radial))

    phasor = area * pol * np.exp(1j * p * theta)
    nominal = np.abs(phasor.sum())
    if nominal <= 0.0:
        return 1.0
    return float(np.abs((r * phasor).sum()) / nominal)


def run_machine_thermal_demag(
    scenario: MachineScenario,
    *,
    i_peak: float,
    turns_per_slot: float,
    gamma_elec: float = 0.0,
    slot_fill: float = 0.45,
    h: float,
    T_amb: float = 20.0,
    dt: float,
    n_steps: int,
    thermal: MachineThermalProperties | None = None,
    T0: float | None = None,
    em_relaxation: float = 0.1,
    em_max_iter: int = 300,
    em_tol: float = 1.0e-6,
    **transient_kwargs,
) -> MachineThermalDemagResult:
    """
    Прогнать сценарий тепловой стойкости магнита на сечении машины.

    Что происходит: обмотка греет машину (медь с ρ(T) — положительная обратная связь),
    температура магнита растёт, колено H_k(T) поднимается, рабочая точка (собственное поле
    + реакция якоря при угле γ) уходит за него — и часть ремнантности теряется НЕОБРАТИМО,
    накапливаясь вдоль траектории. В конце считается, во что это обошлось машине: падение K_e.

    i_peak, gamma_elec, turns_per_slot — режим обмотки; `slot_fill` — коэффициент заполнения
    паза медью (входит в потери, см. шапку модуля). `h`, `T_amb` — конвекция на границах сетки
    (внутренняя расточка + наружная поверхность ротора). `dt`, `n_steps` — горизонт.
    Прочие именованные аргументы уходят в `solve_coupled_magneto_thermal_transient`
    (max_substeps, T_cap, T_bin, demag_relaxation, …).

    ⚠ Границы модели: тепло 2D в сечении (осевой отвод не учитывается — оценка КОНСЕРВАТИВНА
    для короткой машины и оптимистична для длинной), потери в железе пока не входят (нужна
    частота — отдельный инкремент), конвекция одинакова на обеих границах.
    """
    geometry = scenario.geometry
    props = thermal or MachineThermalProperties.representative()
    k_cells, c_cells = props.cell_fields(geometry)

    nu_of_B, nu_init, magnet_mask, _ = machine_reluctivity(geometry, scenario.magnet, scenario.steel)

    # Магнитная задача — МГНОВЕННЫЙ ток при γ; тепловая — СКЗ с поправкой на заполнение паза.
    j_mag = winding_current_density(
        geometry, scenario.layout, i_peak=i_peak, gamma_elec=gamma_elec,
        turns_per_slot=turns_per_slot,
    )
    j_loss = _slot_rms_loss_current(
        geometry, i_peak=i_peak, turns_per_slot=turns_per_slot, slot_fill=slot_fill
    )

    space = LagrangeP1Space2D(geometry.mesh)
    T_start = None if T0 is None else np.full(space.ndofs, float(T0), dtype=float)

    transient = solve_coupled_magneto_thermal_transient(
        space, k_cells=k_cells, capacity_cells=c_cells, h=h, T_amb=T_amb,
        dt=dt, n_steps=n_steps, j_cells=j_mag, j_loss_cells=j_loss,
        magnet=scenario.magnet, magnet_mask=magnet_mask,
        magnet_axis=geometry.magnet_easy_axis, nu_of_B=nu_of_B, nu_init=nu_init,
        T0=T_start, em_relaxation=em_relaxation, em_max_iter=em_max_iter, em_tol=em_tol,
        **transient_kwargs,
    )

    retention = (transient.state.retention.copy() if transient.state is not None
                 else np.ones(int(np.count_nonzero(magnet_mask)), dtype=float))

    return MachineThermalDemagResult(
        transient=transient, retention=retention,
        fundamental_ratio=magnet_fundamental_ratio(geometry, retention, magnet_mask),
        loss_power_initial=float(transient.loss_power[0]),
        loss_power_final=float(transient.loss_power[-1]),
    )
