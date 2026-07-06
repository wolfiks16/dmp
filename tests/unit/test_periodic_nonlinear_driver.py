"""
Драйвер `solve_periodic_nonlinear_mixed_picard` на box-периодике (sub-step 3 §8.7):
(a) линейный предел = периодический MMS через драйвер (якорь периодика+Dirichlet+
калибровка+линейность); (b) насыщающаяся сталь — сходимость, ν активна.
"""
from __future__ import annotations

import numpy as np

from magcore.femcore.manufactured import (
    manufactured_periodic_A,
    manufactured_periodic_rhs,
)
from magcore.femcore.periodic import build_periodic_with_dirichlet, match_periodic_vertices
from magcore.femcore.periodic_nonlinear import solve_periodic_nonlinear_mixed_picard
from magcore.femcore.post import l2_error_at_cell_centroids
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh

TOL = 1e-9
NU = 1.3
_YZ = [(1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)]


def _box_periodic_x_maps(mesh, vs, ss):
    V = mesh.vertices
    n_a = vs.ndofs
    e2d = vs.edge_to_dof_map()

    def onyz(v):
        return any(abs(V[v][ax] - val) < TOL for ax, val in _YZ)

    def edge_in_yz(a, b):
        return any(
            abs(V[a][ax] - val) < TOL and abs(V[b][ax] - val) < TOL for ax, val in _YZ
        )

    dirichlet_global: set[int] = set()
    for v in range(V.shape[0]):
        if onyz(v):
            dirichlet_global.add(n_a + v)
    for (a, b), d in e2d.items():
        if edge_in_yz(a, b):
            dirichlet_global.add(int(d))

    vmap = match_periodic_vertices(
        mesh, lambda c: abs(c[0]) < TOL, lambda c: abs(c[0] - 1.0) < TOL,
        lambda c: np.array([c[0] + 1.0, c[1], c[2]]), tol=TOL,
    )
    return build_periodic_with_dirichlet(vs, ss, vmap, dirichlet_global)


def test_driver_linear_limit_recovers_periodic_manufactured() -> None:
    n = 4
    mesh = build_structured_unit_cube_tetra_mesh(n)
    vs, ss = NedelecP1Space.from_mesh(mesh), LagrangeP1Space(mesh)
    red, dir_red = _box_periodic_x_maps(mesh, vs, ss)

    res = solve_periodic_nonlinear_mixed_picard(
        mesh, vs, ss,
        nu_of_B=lambda B: np.full(mesh.n_cells, NU),
        nu_init=np.full(mesh.n_cells, NU),
        reduction=red, dirichlet_reduced_dofs=dir_red,
        j_fn=lambda x: manufactured_periodic_rhs(x, NU),
        tol=1e-9, max_iter=5,
    )
    assert res.converged
    assert res.n_iterations <= 3  # линейно ⇒ мгновенно
    err = l2_error_at_cell_centroids(vs, res.a, manufactured_periodic_A)
    assert err < 8e-2, err  # восстановил периодическое A_exact (как прямой periodic-MMS)


def test_driver_saturating_steel_converges_periodic() -> None:
    n = 4
    mesh = build_structured_unit_cube_tetra_mesh(n)
    vs, ss = NedelecP1Space.from_mesh(mesh), LagrangeP1Space(mesh)
    red, dir_red = _box_periodic_x_maps(mesh, vs, ss)
    nu0, c = 1.0, 5.0

    def nu_of_B(B):
        return nu0 * (1.0 + c * np.sum(np.asarray(B) ** 2, axis=1))

    res = solve_periodic_nonlinear_mixed_picard(
        mesh, vs, ss, nu_of_B=nu_of_B, nu_init=np.full(mesh.n_cells, nu0),
        reduction=red, dirichlet_reduced_dofs=dir_red,
        j_fn=lambda x: manufactured_periodic_rhs(x, 1.0),
        # под-релаксация: nu↑→|B|↓→nu↓ ⇒ отрицательная производная карты, осцилляция при
        # ω=1 (как demag); ω=0.5 гасит → быстрая сходимость.
        relaxation=0.5, tol=1e-7, max_iter=60,
    )
    assert res.converged
    # нелинейность активна: ν выросла с насыщением и варьируется по ячейкам
    assert res.nu_cells.max() > res.nu_cells.min() + 1e-6
    assert res.nu_cells.max() > nu0 + 1e-3
