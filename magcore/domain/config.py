from __future__ import annotations

from dataclasses import dataclass

from magcore.constants import DEFAULT_LINEAR_TOL
from magcore.enums import BasisKind, FormulationKind, LinearSolverKind
from magcore.domain.validation import ValidationIssue, error


@dataclass(frozen=True, slots=True)
class SolverConfig:
    formulation: FormulationKind = FormulationKind.MULTITRACE
    trial_space_phi: BasisKind = BasisKind.P1
    trial_space_flux: BasisKind = BasisKind.P0
    quadrature_order_regular: int = 4
    quadrature_order_near: int = 6
    linear_solver: LinearSolverKind = LinearSolverKind.DIRECT
    tolerance: float = DEFAULT_LINEAR_TOL
    max_iterations: int | None = None

    def validate_basic(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []

        if self.quadrature_order_regular <= 0:
            issues.append(error("config.quadrature.regular.invalid", "quadrature_order_regular must be > 0.", value=self.quadrature_order_regular))
        if self.quadrature_order_near <= 0:
            issues.append(error("config.quadrature.near.invalid", "quadrature_order_near must be > 0.", value=self.quadrature_order_near))
        if self.tolerance <= 0.0:
            issues.append(error("config.tolerance.invalid", "Solver tolerance must be > 0.", value=self.tolerance))
        if self.max_iterations is not None and self.max_iterations <= 0:
            issues.append(error("config.max_iterations.invalid", "max_iterations must be > 0 when set.", value=self.max_iterations))

        return tuple(issues)