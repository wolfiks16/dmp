from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import solve_thermal
from magcore.hybrid.magnet_demag import DemagRiskMap, MagnetDemagPolicy, compute_demag_risk_map

# Связка магнитостатика↔тепло↔демаг (2D, ядро К6′, полевая форма):
# тепловыделение (потери) → тепловое поле T → рабочая температура магнита T_mag →
# EM-решение с T-зависимым магнитом (Br(T), колено(T)) в приложенном демаг-поле →
# карта риска размагничивания при самосогласованной температуре. Демонстрирует, что
# при данной тепловой нагрузке материал (NdFeB vs SmCo) определяет судьбу магнита.


@dataclass(frozen=True, slots=True)
class MagnetoThermalResult:
    T_field: np.ndarray           # (ndofs,) тепловое поле
    T_magnet: float               # рабочая (hot-spot) температура магнита [°C]
    B_cells: np.ndarray           # (n_cells, 2)
    risk: DemagRiskMap
    em_converged: bool


def _applied_potential_on_boundary(space: LagrangeP1Space2D, B0) -> tuple[np.ndarray, np.ndarray]:
    """Узлы границы и значения A_z^app = B0x·y − B0y·x (однородное фоновое поле B0)."""
    bdofs = np.asarray(space.boundary_dofs(), dtype=int)
    v = space.mesh.vertices[bdofs]
    B0x, B0y = float(B0[0]), float(B0[1])
    vals = B0x * v[:, 1] - B0y * v[:, 0]
    return bdofs, vals


def solve_magneto_thermal_demag(
    space: LagrangeP1Space2D,
    magnet: AnisotropicBHTMagnet,
    magnet_mask: np.ndarray,
    *,
    heat_source_cells: np.ndarray,
    k_cells,
    h: float,
    T_amb: float,
    applied_B0=(0.0, 0.0),
    relaxation: float = 0.5,
    em_max_iter: int = 60,
) -> MagnetoThermalResult:
    """
    Один проход связки тепло→магнит→демаг (ограниченная область + приложенное поле).

    1) Тепло: solve_thermal(k, q=heat_source, Robin h, T_amb) → T-поле; T_mag = hot-spot
       по узлам магнита. 2) EM: планарная задача с T-зависимым магнитом (MagnetDemagPolicy
       при T_mag) в фоновом поле applied_B0 (инхомог. Dirichlet A_z^app). 3) risk-map при T_mag.

    Тепловая сторона верифицирована (2D-T1), EM/демаг — (2D-A/2D-C); здесь — их связка.
    """
    mesh = space.mesh
    nc = mesh.n_cells
    mask = np.asarray(magnet_mask, dtype=bool)

    # 1) Тепловое поле и рабочая температура магнита (hot-spot).
    T_field = solve_thermal(space, k_cells, source=np.asarray(heat_source_cells, dtype=float),
                            h=h, T_amb=T_amb)
    magnet_nodes = np.unique(mesh.cells[mask].reshape(-1))
    T_mag = float(T_field[magnet_nodes].max())

    # 2) EM в приложенном демаг-поле, магнит при T_mag.
    bdofs, app_vals = _applied_potential_on_boundary(space, applied_B0)
    nu_mag = 1.0 / magnet.mu_rec
    nu_cells = np.where(mask, nu_mag, 1.0)
    policy = MagnetDemagPolicy(magnet, mask, T=T_mag, n_cells=nc, axis=(1.0, 0.0))
    em = solve_nonlinear_2d_picard(
        space, nu_of_B=lambda B: nu_cells.copy(), nu_init=nu_cells,
        magnetization=policy, dirichlet_dofs=bdofs, dirichlet_values=app_vals,
        relaxation=relaxation, max_iter=em_max_iter,
    )

    # 3) Карта риска при самосогласованной T_mag.
    risk = compute_demag_risk_map(magnet, em, mask, T=T_mag, axis=(1.0, 0.0))
    return MagnetoThermalResult(
        T_field=T_field, T_magnet=T_mag, B_cells=em.B_cells,
        risk=risk, em_converged=em.converged,
    )
