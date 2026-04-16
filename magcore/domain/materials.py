from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

from magcore.enums import MaterialKind
from magcore.typing import MaterialId, Vector3
from magcore.domain.validation import ValidationIssue, error, warning


def _is_finite_vector3(v: Vector3) -> bool:
    return len(v) == 3 and all(math.isfinite(x) for x in v)


@dataclass(frozen=True, slots=True)
class Material(ABC):
    material_id: MaterialId
    name: str

    @property
    @abstractmethod
    def kind(self) -> MaterialKind:
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> tuple[ValidationIssue, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AirMaterial(Material):
    mu: float

    @property
    def kind(self) -> MaterialKind:
        return MaterialKind.AIR

    def validate(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if not self.material_id.strip():
            issues.append(error("material.id.empty", "Material ID must be non-empty."))
        if not self.name.strip():
            issues.append(error("material.name.empty", "Material name must be non-empty.", material_id=self.material_id))
        if not math.isfinite(self.mu) or self.mu <= 0.0:
            issues.append(error("material.mu.invalid", "Material permeability mu must be finite and > 0.", material_id=self.material_id, mu=self.mu))
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class LinearMagneticMaterial(Material):
    mu: float

    @property
    def kind(self) -> MaterialKind:
        return MaterialKind.LINEAR

    def validate(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if not self.material_id.strip():
            issues.append(error("material.id.empty", "Material ID must be non-empty."))
        if not self.name.strip():
            issues.append(error("material.name.empty", "Material name must be non-empty.", material_id=self.material_id))
        if not math.isfinite(self.mu) or self.mu <= 0.0:
            issues.append(error("material.mu.invalid", "Material permeability mu must be finite and > 0.", material_id=self.material_id, mu=self.mu))
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class PermanentMagnetMaterial(Material):
    mu: float
    br: Vector3

    @property
    def kind(self) -> MaterialKind:
        return MaterialKind.PERMANENT_MAGNET

    def validate(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if not self.material_id.strip():
            issues.append(error("material.id.empty", "Material ID must be non-empty."))
        if not self.name.strip():
            issues.append(error("material.name.empty", "Material name must be non-empty.", material_id=self.material_id))
        if not math.isfinite(self.mu) or self.mu <= 0.0:
            issues.append(error("material.mu.invalid", "Material permeability mu must be finite and > 0.", material_id=self.material_id, mu=self.mu))
        if not _is_finite_vector3(self.br):
            issues.append(error("material.br.invalid", "Remanence vector br must be a finite 3D vector.", material_id=self.material_id, br=self.br))
        elif sum(x * x for x in self.br) == 0.0:
            issues.append(warning("material.br.zero", "Permanent magnet has zero remanence vector.", material_id=self.material_id))
        return tuple(issues)