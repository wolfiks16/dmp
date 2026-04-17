from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_mixed_coulomb_system
from magcore.femcore.boundary_conditions import apply_zero_mixed_dirichlet_bc
from magcore.femcore.gauge_diagnostics import (
    gauge_residual_vector,
    project_to_gradient_subspace,
)
from magcore.femcore.mesh import TetraMesh, oriented_tetra_volume6
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.solver import (
    solve_mixed_coulomb_problem,
    split_mixed_solution,
)
from magcore.femcore.spaces import NedelecP1Space


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
        vol6 = oriented_tetra_volume6(vertices[np.array(tet, dtype=int)])
        if vol6 <= 0.0:
            tet = [a, c, b, 8]
        cells.append(tet)

    return TetraMesh(vertices=vertices, cells=np.asarray(cells, dtype=int))


def _free_scalar_dofs(space: LagrangeP1Space) -> np.ndarray:
    bnd = set(space.boundary_dofs())
    return np.array([i for i in range(space.ndofs) if i not in bnd], dtype=int)


def test_mixed_solver_zero_source_returns_zero_solution() -> None:
    mesh = _cube_with_center_mesh()
    vector_space = NedelecP1Space.from_mesh(mesh)
    scalar_space = LagrangeP1Space(mesh)

    def J_zero(x: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    A, b = assemble_mixed_coulomb_system(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        nu=1.0,
        J_fn=J_zero,
    )

    vector_bnd = vector_space.boundary_dofs()
    scalar_bnd = scalar_space.boundary_dofs()

    A_bc, b_bc = apply_zero_mixed_dirichlet_bc(
        A=A,
        b=b,
        vector_dofs=vector_bnd,
        scalar_dofs=scalar_bnd,
        n_vector_dofs=vector_space.ndofs,
    )

    x = solve_mixed_coulomb_problem(A_bc, b_bc)
    a, p = split_mixed_solution(x, vector_space.ndofs)

    assert np.allclose(x, 0.0, atol=1e-12)
    assert np.allclose(a, 0.0, atol=1e-12)
    assert np.allclose(p, 0.0, atol=1e-12)

    residual = A_bc @ x - b_bc
    assert np.linalg.norm(residual) < 1e-12


def test_mixed_solver_nonzero_source_smoke_and_gauge_checks() -> None:
    mesh = _cube_with_center_mesh()
    vector_space = NedelecP1Space.from_mesh(mesh)
    scalar_space = LagrangeP1Space(mesh)

    def J_fn(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                1.0 + x[0],
                0.5 * x[1],
                x[2],
            ],
            dtype=float,
        )

    A, b = assemble_mixed_coulomb_system(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        nu=1.0,
        J_fn=J_fn,
    )

    vector_bnd = vector_space.boundary_dofs()
    scalar_bnd = scalar_space.boundary_dofs()

    A_bc, b_bc = apply_zero_mixed_dirichlet_bc(
        A=A,
        b=b,
        vector_dofs=vector_bnd,
        scalar_dofs=scalar_bnd,
        n_vector_dofs=vector_space.ndofs,
    )

    x = solve_mixed_coulomb_problem(A_bc, b_bc)
    a, p = split_mixed_solution(x, vector_space.ndofs)

    assert np.isfinite(x).all()
    assert np.isfinite(a).all()
    assert np.isfinite(p).all()

    residual = A_bc @ x - b_bc
    assert np.linalg.norm(residual) < 1e-10

    G = A[:vector_space.ndofs, vector_space.ndofs:]
    G_res = gauge_residual_vector(G, a)
    free_scalar = _free_scalar_dofs(scalar_space)
    assert free_scalar.size > 0
    assert np.linalg.norm(G_res[free_scalar]) < 1e-10

    proj = project_to_gradient_subspace(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        a=a,
    )

    assert np.isfinite(proj.norm_total)
    assert np.isfinite(proj.norm_parallel)
    assert np.isfinite(proj.norm_perp)
    assert np.isfinite(proj.eta_parallel)

    assert proj.eta_parallel < 1e-8