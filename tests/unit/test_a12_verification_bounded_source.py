from __future__ import annotations

import numpy as np

from magcore.femcore.mixed_problem import (
    MixedCoulombProblem,
    MixedCoulombSolution,
    solve_mixed_coulomb_baseline,
)
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _localized_bulk_current(x: np.ndarray) -> np.ndarray:
    xx = float(x[0])
    yy = float(x[1])
    zz = float(x[2])

    s = xx * (1.0 - xx) * yy * (1.0 - yy) * zz * (1.0 - zz)

    return np.array(
        [
            s,
            0.5 * s,
            -0.25 * s,
        ],
        dtype=float,
    )


def _free_scalar_dofs(problem: MixedCoulombProblem) -> np.ndarray:
    boundary = set(problem.scalar_space.boundary_dofs())
    return np.array(
        [i for i in range(problem.n_scalar_dofs) if i not in boundary],
        dtype=int,
    )


def test_a12_bounded_localized_source_solution_is_finite_nontrivial_and_gauge_clean() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)

    solution = solve_mixed_coulomb_baseline(
        mesh=mesh,
        nu=1.0,
        J_fn=_localized_bulk_current,
        compute_gauge_projection=True,
        store_full_system=True,
    )

    assert isinstance(solution, MixedCoulombSolution)

    assert np.isfinite(solution.x).all()
    assert np.isfinite(solution.a).all()
    assert np.isfinite(solution.p).all()

    assert np.linalg.norm(solution.a) > 0.0
    assert solution.norm_total is not None
    assert solution.norm_total > 0.0

    assert solution.linear_residual_norm < 1e-10

    free_scalar = _free_scalar_dofs(solution.problem)
    assert free_scalar.size > 0
    assert np.linalg.norm(solution.gauge_residual[free_scalar]) < 1e-10

    assert solution.gauge_projection is not None
    assert solution.eta_parallel is not None
    assert solution.eta_parallel < 1e-8

    summary = solution.summary_dict()

    assert summary["n_vector_dofs"] == solution.problem.n_vector_dofs
    assert summary["n_scalar_dofs"] == solution.problem.n_scalar_dofs
    assert summary["n_total_dofs"] == solution.problem.n_total_dofs

    assert np.isfinite(float(summary["linear_residual_norm"]))
    assert np.isfinite(float(summary["gauge_residual_norm"]))
    assert np.isfinite(float(summary["eta_parallel"]))
    assert np.isfinite(float(summary["norm_total"]))
    assert np.isfinite(float(summary["norm_parallel"]))
    assert np.isfinite(float(summary["norm_perp"]))


def test_a12_bounded_localized_source_has_positive_curlcurl_energy() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)

    problem = MixedCoulombProblem.from_mesh(
        mesh=mesh,
        nu=1.0,
        J_fn=_localized_bulk_current,
    )

    solution = problem.solve(
        compute_gauge_projection=True,
        store_full_system=True,
    )

    assert solution.system_matrix is not None
    assert solution.rhs is not None

    nA = problem.n_vector_dofs
    K = solution.system_matrix[:nA, :nA]

    assert np.allclose(K, K.T, atol=1e-12)

    energy = float(solution.a @ (K @ solution.a))
    assert np.isfinite(energy)
    assert energy > 0.0

    rhs_vector_block = solution.rhs[:nA]
    assert np.linalg.norm(rhs_vector_block) > 0.0