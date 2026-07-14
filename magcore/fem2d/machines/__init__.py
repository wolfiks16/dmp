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
]
