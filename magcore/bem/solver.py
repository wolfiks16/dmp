from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.domain.config import SolverConfig
from magcore.domain.problem import MagnetostaticProblem
from magcore.bem.assembly import AssembledSystem, assemble_multitrace_system_placeholder


@dataclass(slots=True)
class LinearSolveResult:
    converged: bool
    residual_norm: float
    iterations: int | None
    solver_name: str
    solution_vector: np.ndarray


@dataclass(slots=True)
class SurfaceBEMSolution:
    problem_id: str
    phi_trace_coeffs: np.ndarray
    flux_trace_coeffs: np.ndarray
    solve_result: LinearSolveResult
    config: SolverConfig


def solve_assembled_system(system: AssembledSystem, config: SolverConfig) -> LinearSolveResult:
    x = np.linalg.solve(system.matrix, system.rhs)
    residual = system.matrix @ x - system.rhs
    residual_norm = float(np.linalg.norm(residual))

    solver_name = config.linear_solver.value if hasattr(config.linear_solver, "value") else str(config.linear_solver)

    return LinearSolveResult(
        converged=True,
        residual_norm=residual_norm,
        iterations=None,
        solver_name=solver_name,
        solution_vector=x,
    )


def solve_surface_bem_placeholder(problem: MagnetostaticProblem, config: SolverConfig) -> SurfaceBEMSolution:
    system = assemble_multitrace_system_placeholder(problem, config)
    result = solve_assembled_system(system, config)

    n_phi = system.spaces["phi"].ndofs
    n_flux = system.spaces["flux"].ndofs

    phi = result.solution_vector[:n_phi]
    flux = result.solution_vector[n_phi:n_phi + n_flux]

    return SurfaceBEMSolution(
        problem_id=problem.problem_id,
        phi_trace_coeffs=phi,
        flux_trace_coeffs=flux,
        solve_result=result,
        config=config,
    )