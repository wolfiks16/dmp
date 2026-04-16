from __future__ import annotations

from dataclasses import dataclass
from abc import ABC

from magcore.bem.spaces import DiscreteSpace


@dataclass(frozen=True, slots=True)
class OperatorKind:
    value: str

    SINGLE_LAYER = "single_layer"
    DOUBLE_LAYER = "double_layer"
    ADJOINT_DOUBLE_LAYER = "adjoint_double_layer"
    HYPERSINGULAR = "hypersingular"


@dataclass(frozen=True, slots=True)
class BoundaryOperator(ABC):
    kind: str
    domain_space: DiscreteSpace
    range_space: DiscreteSpace
    dual_space: DiscreteSpace | None
    label: str

    @property
    def shape(self) -> tuple[int, int]:
        return (self.range_space.ndofs, self.domain_space.ndofs)


@dataclass(frozen=True, slots=True)
class SingleLayerOperator(BoundaryOperator):
    pass


@dataclass(frozen=True, slots=True)
class DoubleLayerOperator(BoundaryOperator):
    pass


@dataclass(frozen=True, slots=True)
class AdjointDoubleLayerOperator(BoundaryOperator):
    pass


@dataclass(frozen=True, slots=True)
class HypersingularOperator(BoundaryOperator):
    pass


@dataclass(slots=True)
class AssembledBoundaryOperator:
    kind: str
    matrix: object
    domain_space: DiscreteSpace
    range_space: DiscreteSpace
    label: str
    metadata: dict