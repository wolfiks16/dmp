from __future__ import annotations

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry, Region
from magcore.fem2d.mesh import TriangleMesh
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


def machine_geometry_on_mesh(vertices, cells, region, magnet_easy_axis, slot_id, params) -> MachineGeometry:
    """
    Геометрия машины из готовых массивов — например, из файла расчёта (`fem2d.model.storage`): сетка,
    код региона, ось магнита и номер паза по ячейкам — то, что строит генератор; `params` — его параметры
    (число пазов и полюсов, осевая длина). Ток обмотки по ней считает тот же код, что по построенной.
    """
    mesh = TriangleMesh(vertices=np.asarray(vertices, dtype=float), cells=np.asarray(cells, dtype=int))
    nc = mesh.n_cells
    reg = np.asarray(region, dtype=int).reshape(-1)
    axis = np.asarray(magnet_easy_axis, dtype=float)
    sid = np.asarray(slot_id, dtype=int).reshape(-1)
    if reg.shape != (nc,) or axis.shape != (nc, 2) or sid.shape != (nc,):
        raise ValueError("регион, ось магнита и номер паза — по одному на ячейку сетки.")
    valid = {int(r) for r in Region}
    if not set(np.unique(reg).tolist()) <= valid:
        raise ValueError("в регионах ячеек есть коды, которых у машины нет.")
    if sid.max() >= int(params.n_slots) or sid.min() < -1:
        raise ValueError("номер паза ячейки — от −1 (не паз) до числа пазов − 1.")
    return MachineGeometry(mesh=mesh, region=reg, magnet_easy_axis=axis, slot_id=sid, params=params)
