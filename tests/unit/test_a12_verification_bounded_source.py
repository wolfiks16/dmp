from __future__ import annotations

import numpy as np

from magcore.femcore.mesh import TetraMesh, oriented_tetra_volume6
from magcore.femcore.mixed_problem import (
    MixedCoulombProblem,
    MixedCoulombSolution,
    solve_mixed_coulomb_baseline,
)


def _structured_unit_cube_tet_mesh(n: int) -> TetraMesh:
    """
    Структурированная тетраэдральная сетка единичного куба [0,1]^3
    через разбиение каждого кубика на 6 тетраэдров вдоль главной диагонали.
    """
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


def _localized_bulk_current(x: np.ndarray) -> np.ndarray:
    """
    Гладкий локализованный объёмный ток, затухающий к нулю на границе куба.

    Это физически интерпретируемый bounded-source case для A1.2:
    ток существует внутри объёма, а не навязывается на границе.
    """
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
    """
    Первый физически интерпретируемый bounded verification-case для A1.2.

    Проверяем, что при гладком локализованном объёмном токе:
    1. mixed-задача решается;
    2. решение конечно и нетривиально;
    3. остаток линейной системы мал;
    4. weak/discrete калибровка выполняется;
    5. продольная составляющая остаётся пренебрежимо малой.
    """
    mesh = _structured_unit_cube_tet_mesh(2)

    solution = solve_mixed_coulomb_baseline(
        mesh=mesh,
        nu=1.0,
        J_fn=_localized_bulk_current,
        compute_gauge_projection=True,
        store_full_system=True,
    )

    assert isinstance(solution, MixedCoulombSolution)

    # 1. Решение должно быть конечным
    assert np.isfinite(solution.x).all()
    assert np.isfinite(solution.a).all()
    assert np.isfinite(solution.p).all()

    # 2. Решение должно быть нетривиальным
    assert np.linalg.norm(solution.a) > 0.0
    assert solution.norm_total is not None
    assert solution.norm_total > 0.0

    # 3. Остаток линейной системы должен быть мал
    assert solution.linear_residual_norm < 1e-10

    # 4. Weak/discrete gauge должен выполняться на свободных scalar DOF
    free_scalar = _free_scalar_dofs(solution.problem)
    assert free_scalar.size > 0
    assert np.linalg.norm(solution.gauge_residual[free_scalar]) < 1e-10

    # 5. Продольная часть должна быть пренебрежимо малой
    assert solution.gauge_projection is not None
    assert solution.eta_parallel is not None
    assert solution.eta_parallel < 1e-8

    # 6. В summary все ключевые диагностические величины должны быть конечными
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
    """
    Для нетривиального решения линейной bounded-задачи энергетический блок curl-curl
    должен давать положительную квадратичную форму на найденном векторном решении.
    """
    mesh = _structured_unit_cube_tet_mesh(2)

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

    # Симметрия энергетического блока
    assert np.allclose(K, K.T, atol=1e-12)

    # Энергия curl-curl должна быть положительной для нетривиального решения
    energy = float(solution.a @ (K @ solution.a))
    assert np.isfinite(energy)
    assert energy > 0.0

    # Правая часть должна быть нетривиальной
    rhs_vector_block = solution.rhs[:nA]
    assert np.linalg.norm(rhs_vector_block) > 0.0