from __future__ import annotations

from dataclasses import dataclass
from typing import Union

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
