"""
Параметрические генераторы сечений электрических машин (P1).

Строят реальную геометрию сечения (магнит + сталь + пазы + зазор + обмоточные области)
и выдают `TriangleMesh` ядра + теги областей + ось намагничивания. Мешер — gmsh
(изолирован здесь; решатель его не видит). ⚠ Sergey хочет СВОЙ мешер — вернуться перед
оптимизацией (см. docs/plan_full_2026-07-14.md).
"""
from magcore.fem2d.machines.pmsm_outrunner import (
    MachineGeometry,
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.winding import (
    PHASE_NAMES,
    WindingLayout,
    cell_phase_sign,
    star_of_slots_layout,
)
from magcore.fem2d.machines.excitation import (
    phase_currents,
    slot_areas,
    slot_current_density,
    winding_current_density,
    worst_case_d_axis_currents,
)
from magcore.fem2d.machines.static_solver import (
    MachineStaticResult,
    machine_reluctivity,
    solve_machine_static,
    worst_case_gamma_sweep,
)
from magcore.fem2d.machines.loss import (
    DemagImpact,
    MagnetLossAggregate,
    evaluate_demag_impact,
    magnet_loss_aggregate,
    phase_flux_linkage,
)
from magcore.fem2d.machines.postproc import (
    airgap_cell_mask,
    airgap_torque_arkkio,
    back_emf_constant,
    flux_linkage_amplitude,
)
from magcore.fem2d.machines.operating_point import (
    MagnetOperatingPoint,
    magnet_operating_point,
)
from magcore.fem2d.machines.problem import (
    MachineProblem,
    MachineSolution,
    Scenario,
    solve_machine_problem,
)
from magcore.fem2d.machines.bridge import pmsm_to_problem

__all__ = [
    "MachineGeometry",
    "OutrunnerPMSMParams",
    "Region",
    "build_outrunner_spm_pmsm",
    "PHASE_NAMES",
    "WindingLayout",
    "cell_phase_sign",
    "star_of_slots_layout",
    "phase_currents",
    "slot_areas",
    "slot_current_density",
    "winding_current_density",
    "worst_case_d_axis_currents",
    "MachineStaticResult",
    "machine_reluctivity",
    "solve_machine_static",
    "worst_case_gamma_sweep",
    "DemagImpact",
    "MagnetLossAggregate",
    "evaluate_demag_impact",
    "magnet_loss_aggregate",
    "phase_flux_linkage",
    "airgap_cell_mask",
    "airgap_torque_arkkio",
    "back_emf_constant",
    "flux_linkage_amplitude",
    "MagnetOperatingPoint",
    "magnet_operating_point",
    "MachineProblem",
    "MachineSolution",
    "Scenario",
    "solve_machine_problem",
    "pmsm_to_problem",
]
