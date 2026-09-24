from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.fem2d.model.materials import MagnetMaterial, magnet_groups_of, single_magnet_of
from magcore.fem3d.mesh import TetMesh3D

# ОБЩАЯ регион-объектная постановка 3D-задачи — зеркало `fem2d.model.Problem2D`: сетка + метки
# регионов + материал на регион + ось намагничивания по ячейкам + T. Материалы общие с 2D
# (`fem2d.model.materials`): физический слой один на оба решателя. Решатель — этап 3D-2.


@dataclass(frozen=True, slots=True)
class Region3D:
    """Помеченная область геометрии с назначенным материалом."""

    region_id: int
    name: str
    material: object   # Air | LinearMaterial | SteelMaterial | MagnetMaterial


@dataclass(frozen=True, slots=True)
class Problem3D:
    """
    Постановка 3D-задачи: сетка + метка региона на ячейку + материалы регионов + ось
    намагничивания в ячейках магнита (единичный вектор, вне магнита 0) + температура.
    """

    mesh: TetMesh3D
    cell_region: np.ndarray                # (n_cells,) id региона
    regions: dict                          # region_id -> Region3D
    magnet_axis: np.ndarray | None = None  # (n_cells, 3)
    T: float = 20.0                        # температура [°C]

    def magnet_regions(self) -> list[Region3D]:
        return [r for r in self.regions.values() if isinstance(r.material, MagnetMaterial)]

    def magnet_groups(self) -> list[tuple[AnisotropicBHTMagnet, np.ndarray]]:
        """Магниты задачи по маркам: [(закон марки, маска её ячеек)] — см. `magnet_groups_of`."""
        return magnet_groups_of(self)

    def magnet(self) -> AnisotropicBHTMagnet | None:
        """Закон магнита, когда марка в задаче одна (None — магнитов нет); при нескольких — ошибка."""
        return single_magnet_of(self)

    def magnet_mask(self) -> np.ndarray:
        ids = {r.region_id for r in self.magnet_regions()}
        if not ids:
            return np.zeros(self.mesh.n_cells, dtype=bool)
        return np.isin(self.cell_region, list(ids))

    def region_cell_counts(self) -> np.ndarray:
        n = max(self.regions.keys()) + 1 if self.regions else 0
        return np.bincount(np.asarray(self.cell_region, dtype=np.int64), minlength=n)

    def region_volumes(self) -> np.ndarray:
        """Объёмы регионов [м³] по id региона."""
        n = max(self.regions.keys()) + 1 if self.regions else 0
        return np.bincount(np.asarray(self.cell_region, dtype=np.int64),
                           weights=self.mesh.cell_volumes(), minlength=n)

    def empty_regions(self) -> list[str]:
        """Объекты без ячеек (перекрыты целиком или слишком мелкие) — кроме домена (id 0)."""
        counts = self.region_cell_counts()
        return [r.name for rid, r in sorted(self.regions.items()) if rid != 0 and counts[rid] == 0]

    def validate(self) -> list[str]:
        issues: list[str] = []
        nc = self.mesh.n_cells
        reg = np.asarray(self.cell_region)
        if reg.shape != (nc,):
            issues.append("cell_region должен иметь форму (n_cells,).")
            return issues
        missing = set(np.unique(reg).tolist()) - set(self.regions.keys())
        if missing:
            issues.append(f"нет материала для регионов: {sorted(missing)}.")
        if self.magnet_regions():
            axis = None if self.magnet_axis is None else np.asarray(self.magnet_axis)
            if axis is None or axis.shape != (nc, 3):
                issues.append("для магнитных регионов нужна magnet_axis формы (n_cells, 3).")
            else:
                nrm = np.linalg.norm(axis[self.magnet_mask()], axis=1)
                if np.any((nrm > 1e-12) & (np.abs(nrm - 1.0) > 1e-9)):
                    issues.append("ось намагничивания в ячейках магнита должна быть единичной.")
        return issues

    def check(self) -> None:
        issues = self.validate()
        if issues:
            raise ValueError("Некорректная постановка Problem3D:\n  - " + "\n  - ".join(issues))
