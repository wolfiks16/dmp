from __future__ import annotations

import numpy as np

from magcore.femcore.mesh import TetraMesh, oriented_tetra_volume6
from magcore.femcore.mixed_problem import (
    MixedCoulombProblem,
    MixedCoulombSolution,
    solve_mixed_coulomb_baseline,
)


def _cube_with_center_mesh() -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
            [0.5, 0.5, 0.5],  # 8 center
        ],
        dtype=float,
    )

    boundary_tris = [
        (0, 1, 2), (0, 2, 3),  # z = 0
        (4, 6, 5), (4, 7, 6),  # z = 1
        (0, 5, 1), (0, 4, 5),  # y = 0
        (3, 2, 6), (3, 6, 7),  # y = 1
        (0, 3, 7), (0, 7, 4),  # x = 0
        (1, 5, 6), (1, 6, 2),  # x = 1
    ]

    cells: list[list[int]] = []
    for a, b, c in boundary_tris:
        tet = [a, b, c, 8]
        vol6 = oriented_tetra_volume6(vertices[np.asarray(tet, dtype=int)])
        if vol6 <= 0.0:
            tet = [a, c, b, 8]
        cells.append(tet)

    return TetraMesh(vertices=vertices, cells=np.asarray(cells, dtype=int))


def _free_scalar_dofs(problem: MixedCoulombProblem) -> np.ndarray:
    boundary = set(problem.scalar_space.boundary_dofs())
    return np.array(
        [i for i in range(problem.n_scalar_dofs) if i not in boundary],
        dtype=int,
    )


def test_mixed_problem_zero_source_interface() -> None:
    mesh = _cube_with_center_mesh()

    def J_zero(x: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    solution = solve_mixed_coulomb_baseline(
        mesh=mesh,
        nu=1.0,
        J_fn=J_zero,
        compute_gauge_projection=True,
        store_full_system=True,
    )

    assert isinstance(solution, MixedCoulombSolution)

    assert solution.problem.n_vector_dofs > 0
    assert solution.problem.n_scalar_dofs > 0
    assert solution.problem.n_total_dofs == (
        solution.problem.n_vector_dofs + solution.problem.n_scalar_dofs
    )

    assert np.allclose(solution.x, 0.0, atol=1e-12)
    assert np.allclose(solution.a, 0.0, atol=1e-12)
    assert np.allclose(solution.p, 0.0, atol=1e-12)

    assert solution.linear_residual_norm < 1e-12
    assert solution.gauge_residual_norm < 1e-12

    assert solution.gauge_projection is not None
    assert solution.eta_parallel == 0.0
    assert solution.norm_total == 0.0
    assert solution.norm_parallel == 0.0
    assert solution.norm_perp == 0.0

    assert solution.system_matrix is not None
    assert solution.rhs is not None
    assert solution.system_matrix_bc is not None
    assert solution.rhs_bc is not None


def test_mixed_problem_nonzero_source_interface_and_summary() -> None:
    mesh = _cube_with_center_mesh()

    def J_fn(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                1.0 + x[0],
                0.25 + 0.5 * x[1],
                x[2],
            ],
            dtype=float,
        )

    problem = MixedCoulombProblem.from_mesh(
        mesh=mesh,
        nu=1.0,
        J_fn=J_fn,
    )

    solution = problem.solve(
        compute_gauge_projection=True,
        store_full_system=True,
    )

    assert isinstance(solution, MixedCoulombSolution)

    assert np.isfinite(solution.x).all()
    assert np.isfinite(solution.a).all()
    assert np.isfinite(solution.p).all()

    assert solution.a.shape == (problem.n_vector_dofs,)
    assert solution.p.shape == (problem.n_scalar_dofs,)
    assert solution.x.shape == (problem.n_total_dofs,)

    assert solution.linear_residual.shape == (problem.n_total_dofs,)
    assert solution.gauge_residual.shape == (problem.n_scalar_dofs,)

    assert solution.linear_residual_norm < 1e-10

    free_scalar = _free_scalar_dofs(problem)
    assert free_scalar.size > 0
    assert np.linalg.norm(solution.gauge_residual[free_scalar]) < 1e-10

    assert solution.gauge_projection is not None
    assert solution.eta_parallel is not None
    assert solution.norm_total is not None
    assert solution.norm_parallel is not None
    assert solution.norm_perp is not None

    assert solution.eta_parallel < 1e-8

    summary = solution.summary_dict()

    assert summary["n_vector_dofs"] == problem.n_vector_dofs
    assert summary["n_scalar_dofs"] == problem.n_scalar_dofs
    assert summary["n_total_dofs"] == problem.n_total_dofs

    assert summary["linear_residual_norm"] is not None
    assert summary["gauge_residual_norm"] is not None
    assert summary["eta_parallel"] is not None
    assert summary["norm_total"] is not None
    assert summary["norm_parallel"] is not None
    assert summary["norm_perp"] is not None

    assert float(summary["linear_residual_norm"]) < 1e-10
    assert float(summary["eta_parallel"]) < 1e-8


def test_mixed_problem_boundary_dofs_and_manual_pipeline_match() -> None:
    mesh = _cube_with_center_mesh()

    def J_fn(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                1.0,
                x[1],
                x[2],
            ],
            dtype=float,
        )

    problem = MixedCoulombProblem.from_mesh(
        mesh=mesh,
        nu=2.0,
        J_fn=J_fn,
    )

    vector_bnd, scalar_bnd = problem.boundary_dofs()

    assert len(vector_bnd) > 0
    assert len(scalar_bnd) > 0

    system_matrix, rhs = problem.assemble_system()
    system_matrix_bc, rhs_bc = problem.apply_boundary_conditions(system_matrix, rhs)

    x_manual = np.linalg.solve(system_matrix_bc, rhs_bc)
    solution = problem.solve(
        compute_gauge_projection=False,
        store_full_system=False,
    )

    assert np.allclose(solution.x, x_manual, atol=1e-12)
    assert solution.gauge_projection is None
    assert solution.eta_parallel is None
    assert solution.norm_total is None
    assert solution.norm_parallel is None
    assert solution.norm_perp is None