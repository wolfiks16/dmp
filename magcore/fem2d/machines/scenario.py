from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.bridge import pmsm_to_problem
from magcore.fem2d.machines.excitation import winding_current_density
from magcore.fem2d.machines.loss import phase_flux_linkage
from magcore.fem2d.machines.postproc import back_emf_constant
from magcore.fem2d.machines.pmsm_outrunner import (
    MachineGeometry,
    OutrunnerPMSMParams,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.winding import WindingLayout, star_of_slots_layout
from magcore.fem2d.model import (
    Problem2D,
    Solution2D,
    operating_point,
    solve_problem2d,
    torque_arkkio,
)

# «МАШИНА КАК СЦЕНАРИЙ» поверх общей Problem2D. База (Problem2D + общий пост-проц) не знает про
# ротор/фазы/момент-по-углу — это добавляет СЦЕНАРИЙ машины: обмотка→ток (реакция якоря),
# положение вектора тока (d/q), ЭДС/потокосцепление (обмоточно-взвешенное A_z), момент (общий
# Арккио по зазору). Демонстрирует принцип: физика — в общем ядре, «машинность» — надстройка.


@dataclass(frozen=True, slots=True)
class MachineScenario:
    """Сценарий анализа PMSM: строит Problem2D из обмотки/режима и считает машинные величины."""

    geometry: MachineGeometry
    magnet: AnisotropicBHTMagnet
    steel: SteelBHCurve
    layout: WindingLayout

    @property
    def axial_length(self) -> float:
        return self.geometry.params.axial_length

    # --- постановка и решение через ОБЩУЮ базу ---
    def to_problem(
        self, *, T: float = 20.0, i_peak: float = 0.0, gamma_elec: float = 0.0,
        turns_per_slot: float = 0.0,
    ) -> Problem2D:
        """Собрать общую Problem2D: обмотка+ток (реакция якоря) → j_cells, магнит/сталь/воздух."""
        jz = None
        if i_peak != 0.0 and turns_per_slot != 0.0:
            jz = winding_current_density(
                self.geometry, self.layout, i_peak=i_peak, gamma_elec=gamma_elec,
                turns_per_slot=turns_per_slot,
            )
        return pmsm_to_problem(self.geometry, self.magnet, self.steel, T=T, j_cells=jz)

    def solve(
        self, *, T: float = 20.0, i_peak: float = 0.0, gamma_elec: float = 0.0,
        turns_per_slot: float = 0.0, relaxation: float = 0.1, max_iter: int = 200,
        tol: float = 1.0e-6, track_worst_point: bool = False,
    ) -> Solution2D:
        return solve_problem2d(
            self.to_problem(T=T, i_peak=i_peak, gamma_elec=gamma_elec, turns_per_slot=turns_per_slot),
            relaxation=relaxation, max_iter=max_iter, tol=tol, track_worst_point=track_worst_point,
        )

    # --- машинные величины поверх общего пост-проца ---
    def torque(self, solution: Solution2D) -> float:
        """Электромагнитный момент [Н·м] = общий Арккио по зазорному кольцу."""
        p = self.geometry.params
        return torque_arkkio(solution, p.R_s_out, p.R_mag_in, axial_length=p.axial_length)

    def operating_point(self, solution: Solution2D):
        """Рабочая точка магнита по объёму (общий пост-проц)."""
        return operating_point(solution, axial_length=self.axial_length)

    def phase_flux_linkage(self, solution: Solution2D, *, turns_per_slot: float) -> np.ndarray:
        """Потокосцепление фаз [Вб] (обмоточно-взвешенное A_z; специфично для машины)."""
        return phase_flux_linkage(
            self.geometry, self.layout, solution.field.a, turns_per_slot=turns_per_slot
        )

    def back_emf_constant(self, solution: Solution2D, *, turns_per_slot: float) -> float:
        """ЭДС-постоянная K_e=K_t [В·с/рад] из ХХ-потокосцепления ПМ."""
        return back_emf_constant(
            self.geometry, self.phase_flux_linkage(solution, turns_per_slot=turns_per_slot)
        )

    def torque_vs_current_angle(
        self, gammas, *, i_peak: float, turns_per_slot: float, T: float = 20.0,
        relaxation: float = 0.1, max_iter: int = 200,
    ) -> np.ndarray:
        """
        Свип момента по электрическому углу вектора тока γ (кривая d/q): для каждого γ решить
        общую задачу и взять момент. Момент ∝ i_q ⇒ синусоида по γ (макс на q-оси, ноль на d).
        """
        out = np.empty(len(gammas), dtype=float)
        for i, g in enumerate(gammas):
            sol = self.solve(T=T, i_peak=i_peak, gamma_elec=float(g),
                             turns_per_slot=turns_per_slot, relaxation=relaxation, max_iter=max_iter)
            out[i] = self.torque(sol)
        return out


def machine_scenario(
    params: OutrunnerPMSMParams, magnet: AnisotropicBHTMagnet, steel: SteelBHCurve
) -> MachineScenario:
    """Собрать сценарий: сгенерировать геометрию + сбалансированную обмотку по параметрам."""
    g = build_outrunner_spm_pmsm(params)
    layout = star_of_slots_layout(params.n_slots, params.n_poles)
    return MachineScenario(geometry=g, magnet=magnet, steel=steel, layout=layout)
