from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.fem2d.model.materials import Air, MagnetMaterial
from magcore.fem2d.model.problem import Problem2D, Region2D, solve_problem2d
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import solve_thermal
from magcore.hybrid.magnet_demag import DemagRiskMap

# Связка магнитостатика↔тепло↔демаг (2D, ядро К6′, полевая форма):
# тепловыделение (потери) → тепловое поле T → рабочая температура магнита T_mag →
# EM-решение с T-зависимым магнитом (Br(T), колено(T)) в приложенном демаг-поле →
# карта риска размагничивания при самосогласованной температуре. Демонстрирует, что
# при данной тепловой нагрузке материал (NdFeB vs SmCo) определяет судьбу магнита.


class MagnetOverheatedError(ValueError):
    """
    Рабочая температура магнита вышла за предел валидности модели (перегрев/разгон).
    Несёт T_magnet, limit и уже посчитанное T_field — чтобы вызывающий (пилот) мог
    показать температурное поле и дать рекомендацию, не пересчитывая тепло.
    """

    def __init__(self, message: str, *, T_magnet: float, limit: float, T_field):
        super().__init__(message)
        self.T_magnet = float(T_magnet)
        self.limit = float(limit)
        self.T_field = T_field


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
    method: str = "newton",
) -> MagnetoThermalResult:
    """
    Один проход связки тепло→магнит→демаг (ограниченная область + приложенное поле).

    1) Тепло: solve_thermal(k, q=heat_source, Robin h, T_amb) → T-поле; T_mag = hot-spot
       по узлам магнита. 2) EM: планарная задача с T-зависимым магнитом при T_mag (ось (1,0)) в
       фоновом поле applied_B0 (инхомог. Dirichlet A_z^app) — общий `solve_problem2d`: по умолчанию
       магнит законом ветви в касательной Ньютона; `method='picard'` — прежняя схема (источник с
       релаксацией `relaxation`, за коленом не сходится — Л-107), эталон. 3) risk-map при T_mag.

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

    # Защита: T_mag за пределом валидности модели магнита ⇒ понятная ошибка вместо
    # криптичного отказа в curve_at (магнит «сварен» — тепловой разгон / потеря свойств).
    limit = magnet.temperature_limit()
    if T_mag >= limit:
        raise MagnetOverheatedError(
            "Магнит перегрет: T=%.0f C >= предел модели %.0f C (тепловой разгон / "
            "потеря свойств). Снизьте тепловую нагрузку или усильте охлаждение."
            % (T_mag, limit),
            T_magnet=T_mag, limit=limit, T_field=T_field,
        )

    # 2) EM в приложенном демаг-поле, магнит при T_mag: воздух + магнит с осью (1,0) — общая задача.
    bdofs, app_vals = _applied_potential_on_boundary(space, applied_B0)
    axis = np.zeros((nc, 2), dtype=float)
    axis[mask] = (1.0, 0.0)
    problem = Problem2D(
        mesh=mesh, cell_region=mask.astype(int),
        regions={0: Region2D(0, "air", Air()), 1: Region2D(1, "magnet", MagnetMaterial(magnet))},
        magnet_axis=axis, T=T_mag, dirichlet_dofs=bdofs, dirichlet_values=app_vals,
    )
    sol = solve_problem2d(problem, method=method, relaxation=relaxation, max_iter=em_max_iter)

    # 3) Карта риска при самосогласованной T_mag (посчитана общим решателем по той же оси).
    return MagnetoThermalResult(
        T_field=T_field, T_magnet=T_mag, B_cells=sol.field.B_cells,
        risk=sol.risk, em_converged=sol.field.converged,
    )
