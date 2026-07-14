from __future__ import annotations

import math

import numpy as np

from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry
from magcore.fem2d.machines.winding import WindingLayout

# P3: поле реакции якоря = то, которое обмотка РЕАЛЬНО создаёт по своим параметрам
# (число проводников в пазу N + фазные токи + раскладка фаз), а не заданное вручную число.
# Здесь — только источник тока J_z (плотность А/м² по нормали к плоскости) из параметров
# катушки; само поле B решается в P4. worst-case берём ПО-ПРОСТОМУ: амплитуда + d-ось
# (демагнитизирующий ток); ТОЧНЫЙ худший электрический угол ищется свипом в P4 (нужно
# решённое поле, чтобы минимизировать H_par в магните).


def phase_currents(i_peak: float, gamma_elec: float) -> np.ndarray:
    """
    Мгновенные фазные токи [I_A, I_B, I_C] сбалансированной 3-фазной системы для вектора
    тока с амплитудой `i_peak` и электрическим углом `gamma_elec` [рад]:
        I_A = I cos(γ),  I_B = I cos(γ − 2π/3),  I_C = I cos(γ − 4π/3).
    Сумма = 0 (нет нулевой последовательности). Ось/направление «реакции якоря» задаёт γ.
    """
    g = float(gamma_elec)
    return float(i_peak) * np.array([
        math.cos(g),
        math.cos(g - 2.0 * math.pi / 3.0),
        math.cos(g - 4.0 * math.pi / 3.0),
    ])


def worst_case_d_axis_currents(i_peak: float) -> np.ndarray:
    """
    Простой worst-case для демага: чисто демагнитизирующий d-ток (I_d = −i_peak, I_q = 0)
    при совмещении оси фазы A с осью магнита ⇒ γ = π ⇒ [−I, I/2, I/2]. МДС статора
    направлена ПРОТИВ магнита. ⚠ упрощение: ТОЧНЫЙ худший угол γ (и положение ротора)
    определяется свипом в P4 по минимуму H_par в магните — здесь фиксируем разумный режим.
    """
    return phase_currents(i_peak, math.pi)


def _cell_areas(geometry: MachineGeometry) -> np.ndarray:
    mesh = geometry.mesh
    return np.array([mesh.cell_area(c) for c in range(mesh.n_cells)], dtype=float)


def slot_areas(geometry: MachineGeometry) -> np.ndarray:
    """Площадь меди каждого паза (сумма площадей его ячеек), (n_slots,)."""
    n = geometry.params.n_slots
    areas = np.zeros(n, dtype=float)
    sid = geometry.slot_id
    cell_area = _cell_areas(geometry)
    in_slot = sid >= 0
    np.add.at(areas, sid[in_slot], cell_area[in_slot])
    return areas


def slot_current_density(
    geometry: MachineGeometry,
    layout: WindingLayout,
    phase_current: np.ndarray,
    turns_per_slot: float,
) -> np.ndarray:
    """
    Поячеечная плотность тока J_z (n_cells,) от обмотки. В ячейке паза s:
        J_z = sign_s · N · I_{phase_s} / A_slot(s),
    так что ∫_s J_z dA = sign_s · N · I_{phase_s} — ровно нужные А·витки со знаком фазы
    (равномерно размазаны по сечению паза). Вне пазов 0. N = `turns_per_slot` (число
    проводников/витков в пазу), `phase_current` = [I_A, I_B, I_C].
    """
    if layout.n_slots != geometry.params.n_slots:
        raise ValueError("layout.n_slots не совпадает с геометрией.")
    pc = np.asarray(phase_current, dtype=float).reshape(-1)
    if pc.shape != (3,):
        raise ValueError("phase_current должен быть [I_A, I_B, I_C].")

    sid = geometry.slot_id
    A = slot_areas(geometry)
    jz = np.zeros(geometry.mesh.n_cells, dtype=float)
    in_slot = sid >= 0
    s = sid[in_slot]
    ph = layout.phase_of_slot[s]
    sg = layout.sign_of_slot[s].astype(float)
    jz[in_slot] = sg * float(turns_per_slot) * pc[ph] / A[s]
    return jz


def winding_current_density(
    geometry: MachineGeometry,
    layout: WindingLayout,
    *,
    i_peak: float,
    gamma_elec: float,
    turns_per_slot: float,
) -> np.ndarray:
    """Удобная связка: (амплитуда, эл. угол γ) → фазные токи → поячеечный J_z (P3→P4)."""
    return slot_current_density(
        geometry, layout, phase_currents(i_peak, gamma_elec), turns_per_slot
    )
