from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import (
    assemble_coulomb_coupling_matrix,
    assemble_mixed_coulomb_system,
)
from magcore.femcore.boundary_conditions import apply_zero_mixed_dirichlet_bc
from magcore.femcore.gauge_diagnostics import (
    gauge_residual_vector,
    project_to_gradient_subspace,
)
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.solver import (
    solve_mixed_coulomb_problem,
    split_mixed_solution,
)
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh import TetraMesh, oriented_tetra_volume6


def _structured_unit_cube_tet_mesh(n: int) -> TetraMesh:
    if n < 1:
        raise ValueError("n must be at least 1.")

    h = 1.0 / n

    def vid(i: int, j: int, k: int) -> int:
        return i + (n + 1) * (j + (n + 1) * k)

    vertices = []
    for k in range(n + 1):
        z = k * h
        for j in range(n + 1):
            y = j * h
            for i in range(n + 1):
                x = i * h
                vertices.append([x, y, z])
    vertices = np.asarray(vertices, dtype=float)

    cells: list[list[int]] = []

    for k in range(n):
        for j in range(n):
            for i in range(n):
                v000 = vid(i, j, k)
                v100 = vid(i + 1, j, k)
                v010 = vid(i, j + 1, k)
                v110 = vid(i + 1, j + 1, k)
                v001 = vid(i, j, k + 1)
                v101 = vid(i + 1, j, k + 1)
                v011 = vid(i, j + 1, k + 1)
                v111 = vid(i + 1, j + 1, k + 1)

                local_tets = [
                    [v000, v100, v110, v111],
                    [v000, v100, v101, v111],
                    [v000, v001, v101, v111],
                    [v000, v001, v011, v111],
                    [v000, v010, v011, v111],
                    [v000, v010, v110, v111],
                ]

                for tet in local_tets:
                    vol6 = oriented_tetra_volume6(vertices[np.asarray(tet, dtype=int)])
                    if vol6 <= 0.0:
                        tet = [tet[0], tet[2], tet[1], tet[3]]
                    cells.append(tet)

    return TetraMesh(vertices=vertices, cells=np.asarray(cells, dtype=int))


def _free_vector_dofs(space: NedelecP1Space) -> np.ndarray:
    boundary = set(space.boundary_dofs())
    return np.array([i for i in range(space.ndofs) if i not in boundary], dtype=int)


def _free_scalar_dofs(space: LagrangeP1Space) -> np.ndarray:
    boundary = set(space.boundary_dofs())
    return np.array([i for i in range(space.ndofs) if i not in boundary], dtype=int)


def _nullspace(matrix: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("matrix must be 2D.")

    _, s, vt = np.linalg.svd(M, full_matrices=True)

    if s.size == 0:
        return np.eye(M.shape[1], dtype=float)

    tol = rtol * max(M.shape) * s[0]
    rank = int(np.sum(s > tol))
    return vt[rank:].T.copy()


def _build_exact_discrete_state(
    mesh: TetraMesh,
    vector_space: NedelecP1Space,
    scalar_space: LagrangeP1Space,
) -> tuple[np.ndarray, np.ndarray]:
    free_vec = _free_vector_dofs(vector_space)
    free_sca = _free_scalar_dofs(scalar_space)

    if free_sca.size == 0:
        raise ValueError("This test requires at least one free scalar DOF. Use n >= 2.")
    if free_vec.size == 0:
        raise ValueError("This test requires at least one free vector DOF.")

    G = assemble_coulomb_coupling_matrix(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        quadrature_order=2,
    )

    G_free = G[np.ix_(free_vec, free_sca)]

    null_basis = _nullspace(G_free.T)
    if null_basis.shape[1] == 0:
        raise AssertionError(
            "No nontrivial nullspace found for G_free^T; "
            "this should not happen on the chosen mesh sequence."
        )

    a_free = null_basis[:, 0].copy()
    a_free /= np.linalg.norm(a_free)

    coords = mesh.vertices[free_sca]
    p_free = 1.0 + coords[:, 0] + 2.0 * coords[:, 1] + 3.0 * coords[:, 2]

    a_exact = np.zeros(vector_space.ndofs, dtype=float)
    p_exact = np.zeros(scalar_space.ndofs, dtype=float)

    a_exact[free_vec] = 0.75 * a_free
    p_exact[free_sca] = 0.25 * p_free

    return a_exact, p_exact


def _solve_discrete_manufactured_case(n: int) -> dict[str, float | np.ndarray]:
    mesh = _structured_unit_cube_tet_mesh(n)
    vector_space = NedelecP1Space.from_mesh(mesh)
    scalar_space = LagrangeP1Space(mesh)

    def J_zero(x: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    A, _ = assemble_mixed_coulomb_system(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        nu=1.0,
        J_fn=J_zero,
    )

    nA = vector_space.ndofs
    K = A[:nA, :nA]
    G = A[:nA, nA:]

    a_exact, p_exact = _build_exact_discrete_state(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
    )

    f_exact = K @ a_exact + G @ p_exact

    b = np.zeros(nA + scalar_space.ndofs, dtype=float)
    b[:nA] = f_exact

    vector_bnd = vector_space.boundary_dofs()
    scalar_bnd = scalar_space.boundary_dofs()

    A_bc, b_bc = apply_zero_mixed_dirichlet_bc(
        A=A,
        b=b,
        vector_dofs=vector_bnd,
        scalar_dofs=scalar_bnd,
        n_vector_dofs=nA,
    )

    x = solve_mixed_coulomb_problem(A_bc, b_bc)
    a_num, p_num = split_mixed_solution(x, nA)

    x_exact = np.concatenate([a_exact, p_exact])

    residual_norm = float(np.linalg.norm(A_bc @ x - b_bc))
    error_norm = float(np.linalg.norm(x - x_exact))

    gauge_vec = gauge_residual_vector(G, a_num)
    free_sca = _free_scalar_dofs(scalar_space)
    gauge_free_norm = float(np.linalg.norm(gauge_vec[free_sca]))

    proj = project_to_gradient_subspace(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        a=a_num,
    )

    return {
        "mesh": mesh,
        "vector_space": vector_space,
        "scalar_space": scalar_space,
        "solution": x,
        "a_num": a_num,
        "p_num": p_num,
        "a_exact": a_exact,
        "p_exact": p_exact,
        "x_exact": x_exact,
        "residual_norm": residual_norm,
        "error_norm": error_norm,
        "gauge_free_norm": gauge_free_norm,
        "eta_parallel": float(proj.eta_parallel),
    }


def test_a12_discrete_manufactured_state_is_recovered_on_refined_meshes() -> None:
    results = [
        _solve_discrete_manufactured_case(2),
        _solve_discrete_manufactured_case(3),
    ]

    for result in results:
        assert np.isfinite(result["solution"]).all()
        assert np.isfinite(result["a_num"]).all()
        assert np.isfinite(result["p_num"]).all()

        assert result["residual_norm"] < 1e-10

        assert result["error_norm"] < 1e-10
        assert np.allclose(result["a_num"], result["a_exact"], atol=1e-10)
        assert np.allclose(result["p_num"], result["p_exact"], atol=1e-10)

        assert result["gauge_free_norm"] < 1e-10
        assert result["eta_parallel"] < 1e-10