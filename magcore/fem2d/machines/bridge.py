from __future__ import annotations

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry, Region
from magcore.fem2d.model.materials import Air, MagnetMaterial, SteelMaterial
from magcore.fem2d.model.problem import Problem2D, Region2D

# Мост: параметрический PMSM-генератор → ОБЩАЯ Problem2D. Тем самым PMSM превращается из
# «зашитой модели» в один из ПОСТАВЩИКОВ общей задачи — решатель и пост-проц работают с
# Problem2D, а не с MachineGeometry. Сталь → зубцы/ярма, магнит → кольцо магнитов (ось —
# поячеечная радиальная), воздух → зазор/пазы. Ток (реакция якоря) передаётся поячеечно.


def pmsm_to_problem(
    geometry: MachineGeometry,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
    *,
    T: float = 20.0,
    j_cells: np.ndarray | None = None,
) -> Problem2D:
    """Преобразовать сгенерированную геометрию outrunner PMSM в общую Problem2D."""
    regions = {
        int(Region.AIR_GAP): Region2D(int(Region.AIR_GAP), "air_gap", Air()),
        int(Region.SLOT): Region2D(int(Region.SLOT), "slot", Air()),        # медь — немагнитна
        int(Region.STATOR_YOKE): Region2D(int(Region.STATOR_YOKE), "stator_yoke", SteelMaterial(steel)),
        int(Region.TOOTH): Region2D(int(Region.TOOTH), "tooth", SteelMaterial(steel)),
        int(Region.ROTOR_YOKE): Region2D(int(Region.ROTOR_YOKE), "rotor_yoke", SteelMaterial(steel)),
        int(Region.MAGNET): Region2D(int(Region.MAGNET), "magnet", MagnetMaterial(magnet)),
    }
    return Problem2D(
        mesh=geometry.mesh,
        cell_region=geometry.region,
        regions=regions,
        magnet_axis=geometry.magnet_easy_axis,
        j_cells=j_cells,
        T=float(T),
    )
