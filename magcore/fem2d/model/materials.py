from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve

# Материалы ОБЩЕЙ 2D-модели задачи: назначаются региону геометрии независимо от того, что
# это за конструкция (мотор, ДП, произвольная задача). Все — данные; решатель разбирает их
# в поячеечную релуктивность/источник. Конвенция ν — ОТНОСИТЕЛЬНАЯ (1/μ_r), как во всём fem2d.


@dataclass(frozen=True, slots=True)
class Air:
    """Немагнитная область (воздух/медь-паз/зазор): относительная ν = 1."""
    name: str = "air"


@dataclass(frozen=True, slots=True)
class LinearMaterial:
    """Линейный магнитный материал: постоянная относительная проницаемость μ_r (ν = 1/μ_r)."""
    mu_r: float
    name: str = "linear"

    def __post_init__(self) -> None:
        if not (self.mu_r > 0.0):
            raise ValueError("mu_r must be positive.")


@dataclass(frozen=True, slots=True)
class SteelMaterial:
    """Нелинейная сталь по кривой B(H) (насыщение); ν = μ₀·ν_chord(|B|) = 1/μ_r(|B|)."""
    curve: SteelBHCurve
    name: str = "steel"


@dataclass(frozen=True, slots=True)
class MagnetMaterial:
    """
    Постоянный магнит B(H,T) с необратимым коленом. ν = 1/μ_rec (recoil, const);
    источник ν·B_r_eff·e через MagnetDemagPolicy. Ось лёгкого намагничивания — ПОЯЧЕЕЧНАЯ
    (задаётся в Problem2D.magnet_axis), т.к. в реальной геометрии она разная (напр. радиальная).
    """
    magnet: AnisotropicBHTMagnet
    name: str = "magnet"


Material = Union[Air, LinearMaterial, SteelMaterial, MagnetMaterial]


def magnet_groups_of(problem) -> list[tuple[AnisotropicBHTMagnet, np.ndarray]]:
    """
    Магниты постановки (2D или 3D: нужны `regions` и `cell_region`) по маркам — [(закон марки, маска её
    ячеек)] в порядке первого региона марки. Марка — закон материала (`AnisotropicBHTMagnet.law_key`), а не
    объект в памяти: регионы из одинакового материала, созданного порознь, — одна группа и один закон.
    Марка без ячеек (регион перекрыт другим) остаётся в списке с пустой маской.
    """
    groups: dict[tuple, tuple[AnisotropicBHTMagnet, list[int]]] = {}
    for rid, region in sorted(problem.regions.items()):
        if isinstance(region.material, MagnetMaterial):
            m = region.material.magnet
            groups.setdefault(m.law_key(), (m, []))[1].append(rid)
    reg = np.asarray(problem.cell_region)
    return [(m, np.isin(reg, ids)) for m, ids in groups.values()]


def single_magnet_of(problem) -> AnisotropicBHTMagnet | None:
    """
    Закон магнита постановки, когда марка в ней одна; None — магнитов нет. При нескольких марках одного
    закона на всю задачу нет — ошибка с их перечнем: такому коду нужно считать по маркам (`magnet_groups_of`).
    """
    groups = magnet_groups_of(problem)
    if len(groups) > 1:
        raise ValueError("в задаче несколько марок магнита (" + ", ".join(m.name for m, _ in groups)
                         + ") — одного закона на всю задачу нет, считать нужно по маркам (magnet_groups).")
    return groups[0][0] if groups else None
