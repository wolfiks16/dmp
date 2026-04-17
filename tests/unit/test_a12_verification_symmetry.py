from __future__ import annotations

import numpy as np

from magcore.femcore.mesh import TetraMesh, oriented_tetra_volume6
from magcore.femcore.mixed_problem import (
    MixedCoulombSolution,
    solve_mixed_coulomb_baseline,
)
from magcore.femcore.spaces import NedelecP1Space


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


def _symmetric_bulk_current_about_x_midplane(x: np.ndarray) -> np.ndarray:
    """
    Гладкий объёмный ток, симметричный относительно плоскости x = 0.5.

    Для отражения R(x, y, z) = (1 - x, y, z) берём полярный векторный закон симметрии:
        J(x) = S J(Rx),   S = diag(-1, 1, 1)

    Здесь источник выбран как
        J = (0, s, 0),
    где s(1-x, y, z) = s(x, y, z), поэтому условие симметрии выполнено.
    """
    xx = float(x[0])
    yy = float(x[1])
    zz = float(x[2])

    s = xx * (1.0 - xx) * yy * (1.0 - yy) * zz * (1.0 - zz)

    return np.array([0.0, s, 0.0], dtype=float)


def _vertex_key(x: np.ndarray, decimals: int = 12) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(x, dtype=float), decimals=decimals))


def _build_vertex_reflection_map_x_midplane(mesh: TetraMesh) -> np.ndarray:
    """
    Для каждой вершины i вернуть индекс вершины j, являющейся отражением
    относительно плоскости x = 0.5:
        (x, y, z) -> (1-x, y, z)
    """
    lookup = {
        _vertex_key(mesh.vertices[i]): i
        for i in range(mesh.n_vertices)
    }

    reflection = np.zeros(mesh.n_vertices, dtype=int)

    for i, xyz in enumerate(mesh.vertices):
        xr = np.array([1.0 - xyz[0], xyz[1], xyz[2]], dtype=float)
        key = _vertex_key(xr)
        if key not in lookup:
            raise AssertionError("Reflected vertex not found in mesh.")
        reflection[i] = lookup[key]

    return reflection


def _build_edge_reflection_map_x_midplane(
    vector_space: NedelecP1Space,
    vertex_reflection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Для каждого глобального edge-DOF вернуть:
    - индекс отражённого DOF,
    - знак, возникающий из-за возможного переворота канонической ориентации.

    Если e = (i, j), i < j, то его отражение как ориентированного ребра есть
    (R(i), R(j)). Хранимый глобальный DOF всегда привязан к канонической
    ориентации min/max, поэтому нужен знак.
    """
    edge_to_dof = vector_space.edge_to_dof_map()

    reflected_dof = np.zeros(vector_space.ndofs, dtype=int)
    reflected_sign = np.zeros(vector_space.ndofs, dtype=int)

    for dof_idx, (i, j) in enumerate(vector_space.global_edges):
        ri = int(vertex_reflection[i])
        rj = int(vertex_reflection[j])

        if ri < rj:
            reflected_edge = (ri, rj)
            sign = +1
        else:
            reflected_edge = (rj, ri)
            sign = -1

        if reflected_edge not in edge_to_dof:
            raise AssertionError("Reflected edge not found in global edge set.")

        reflected_dof[dof_idx] = edge_to_dof[reflected_edge]
        reflected_sign[dof_idx] = sign

    return reflected_dof, reflected_sign


def _free_scalar_dofs(solution: MixedCoulombSolution) -> np.ndarray:
    boundary = set(solution.problem.scalar_space.boundary_dofs())
    return np.array(
        [i for i in range(solution.problem.n_scalar_dofs) if i not in boundary],
        dtype=int,
    )


def test_a12_solution_respects_x_midplane_symmetry_for_symmetric_source() -> None:
    """
    Verification-case на симметрию для bounded A1.2.

    Проверяем, что при симметричном относительно плоскости x = 0.5 источнике:
    1. mixed-задача решается;
    2. residual и gauge diagnostics малы;
    3. скалярный множитель p_h симметричен;
    4. векторный потенциал A_h удовлетворяет дискретной отражательной симметрии
       на уровне edge-DOF с учётом канонической ориентации рёбер.
    """
    mesh = _structured_unit_cube_tet_mesh(3)

    solution = solve_mixed_coulomb_baseline(
        mesh=mesh,
        nu=1.0,
        J_fn=_symmetric_bulk_current_about_x_midplane,
        compute_gauge_projection=True,
        store_full_system=False,
    )

    assert isinstance(solution, MixedCoulombSolution)

    # Базовая корректность решения
    assert np.isfinite(solution.x).all()
    assert np.isfinite(solution.a).all()
    assert np.isfinite(solution.p).all()
    assert solution.linear_residual_norm < 1e-10

    free_scalar = _free_scalar_dofs(solution)
    assert free_scalar.size > 0
    assert np.linalg.norm(solution.gauge_residual[free_scalar]) < 1e-10

    assert solution.eta_parallel is not None
    assert solution.eta_parallel < 1e-8

    # Строим дискретное отражение
    vertex_reflection = _build_vertex_reflection_map_x_midplane(mesh)
    edge_reflection_dof, edge_reflection_sign = _build_edge_reflection_map_x_midplane(
        solution.problem.vector_space,
        vertex_reflection,
    )

    # --- Проверка симметрии scalar unknown p_h ---
    #
    # Для скаляра ожидаем:
    #   p(Rx) = p(x)
    #
    p_reflected = solution.p[vertex_reflection]

    p_norm = float(np.linalg.norm(solution.p))
    p_sym_abs = float(np.linalg.norm(p_reflected - solution.p))

    if p_norm > 1e-14:
        p_sym_rel = p_sym_abs / p_norm
        assert p_sym_rel < 1e-8
    else:
        assert p_sym_abs < 1e-12

    # --- Проверка симметрии vector unknown A_h ---
    #
    # Для полярного вектора при отражении относительно x = 0.5:
    #   A(x) = S A(Rx),   S = diag(-1, 1, 1)
    #
    # На уровне edge-DOF это даёт:
    #   a_reflected(edge') = sign(edge -> edge') * a(edge)
    #
    a_reflected_expected = edge_reflection_sign.astype(float) * solution.a
    a_reflected_actual = solution.a[edge_reflection_dof]

    a_norm = float(np.linalg.norm(solution.a))
    a_sym_abs = float(np.linalg.norm(a_reflected_actual - a_reflected_expected))

    assert a_norm > 0.0
    a_sym_rel = a_sym_abs / a_norm
    assert a_sym_rel < 1e-8