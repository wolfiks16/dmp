"""
Общая регион-объектная модель 2D-задачи (носитель универсальности комплекса).

`Problem2D` = произвольная сетка + помеченные регионы + материал на регион + ток + ГУ →
общий `solve_problem2d`. Не привязана к конкретной конструкции; PMSM и др. — поставщики
такой задачи. Физика/решатель переиспользуются из верифицированного ядра fem2d.
"""
from magcore.fem2d.model.materials import (
    Air,
    LinearMaterial,
    MagnetMaterial,
    Material,
    SteelMaterial,
)
from magcore.fem2d.model.problem import (
    Problem2D,
    Region2D,
    Solution2D,
    solve_problem2d,
)
from magcore.fem2d.model.postproc import (
    OperatingPointField,
    flux_between_points,
    interpolate_Az,
    magnetic_energy,
    operating_point,
    torque_arkkio,
)

__all__ = [
    "Air",
    "LinearMaterial",
    "SteelMaterial",
    "MagnetMaterial",
    "Material",
    "Problem2D",
    "Region2D",
    "Solution2D",
    "solve_problem2d",
    "OperatingPointField",
    "flux_between_points",
    "interpolate_Az",
    "magnetic_energy",
    "operating_point",
    "torque_arkkio",
]
