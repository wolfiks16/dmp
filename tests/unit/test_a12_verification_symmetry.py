from __future__ import annotations

import numpy as np

from magcore.femcore.mixed_problem import (
    MixedCoulombSolution,
    solve_mixed_coulomb_baseline,
)
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh import TetraMesh
from magcore.mesh.mesh_generators import build_symmetric_unit_cube_tetra_mesh


def _symmetric_bulk_current_about_x_midplane(x: np.ndarray) -> np.ndarray:
    xx = float(x[0])
    yy = float(x[1])
    zz = float(x[2])

    s = xx * (1.0 - xx) * yy * (1.0 - yy) * zz * (1.0 - zz)
    return np.array([0.0, s, 0.0], dtype=float)


def _vertex_key(x: np.ndarray, decimals: int = 12) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(x, dtype=float), decimals=decimals))


def _build_vertex_reflection_map_x_midplane(mesh: TetraMesh) -> np.ndarray:
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
    mesh = build_symmetric_unit_cube_tetra_mesh()

    solution = solve_mixed_coulomb_baseline(
        mesh=mesh,
        nu=1.0,
        J_fn=_symmetric_bulk_current_about_x_midplane,
        compute_gauge_projection=True,
        store_full_system=False,
    )

    assert isinstance(solution, MixedCoulombSolution)

    assert np.isfinite(solution.x).all()
    assert np.isfinite(solution.a).all()
    assert np.isfinite(solution.p).all()
    assert solution.linear_residual_norm < 1e-10

    free_scalar = _free_scalar_dofs(solution)
    assert free_scalar.size > 0
    assert np.linalg.norm(solution.gauge_residual[free_scalar]) < 1e-10

    assert solution.eta_parallel is not None
    assert solution.eta_parallel < 1e-8

    vertex_reflection = _build_vertex_reflection_map_x_midplane(mesh)
    edge_reflection_dof, edge_reflection_sign = _build_edge_reflection_map_x_midplane(
        solution.problem.vector_space,
        vertex_reflection,
    )

    p_reflected = solution.p[vertex_reflection]
    p_norm = float(np.linalg.norm(solution.p))
    p_sym_abs = float(np.linalg.norm(p_reflected - solution.p))

    if p_norm > 1e-14:
        p_sym_rel = p_sym_abs / p_norm
        assert p_sym_rel < 1e-8
    else:
        assert p_sym_abs < 1e-12

    a_reflected_expected = edge_reflection_sign.astype(float) * solution.a
    a_reflected_actual = solution.a[edge_reflection_dof]

    a_norm = float(np.linalg.norm(solution.a))
    a_sym_abs = float(np.linalg.norm(a_reflected_actual - a_reflected_expected))

    assert a_norm > 0.0
    a_sym_rel = a_sym_abs / a_norm
    assert a_sym_rel < 1e-8