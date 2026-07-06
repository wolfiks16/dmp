"""
Периодический MMS — строгая верификация (анти)периодических ГУ на РЕАЛЬНОМ решении.

Manufactured A=(0, cos(2πx)·z(1−z), 0): x-периодично (период 1), tangential A=0 на
гранях y,z, div A=0 ⇒ p=0; источник J из `manufactured_periodic_rhs`. Решатель с
периодикой-x + Dirichlet-yz ОБЯЗАН восстановить A_exact с порядком ~1. Угловые DOF
(пересечение периодической и Dirichlet-граней) отдаём Dirichlet (там циркуляция=0).
"""
from __future__ import annotations

import math

import numpy as np

from magcore.femcore.assembly import assemble_mixed_coulomb_system
from magcore.femcore.manufactured import (
    manufactured_periodic_A,
    manufactured_periodic_curl,
    manufactured_periodic_rhs,
)
from magcore.femcore.periodic import (
    build_periodic_reduction,
    expand_solution,
    match_periodic_vertices,
    reduce_system,
)
from magcore.femcore.post import (
    l2_curl_error_at_cell_centroids,
    l2_error_at_cell_centroids,
)
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh

TOL = 1e-9
NU = 1.3


def _f(c, axis, val):
    return abs(c[axis] - val) < TOL


def _solve_periodic_mms(n: int):
    mesh = build_structured_unit_cube_tetra_mesh(n)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    n_a = vs.ndofs
    V = mesh.vertices
    e2d = vs.edge_to_dof_map()

    # сборка смешанной системы с периодическим источником
    # (RHS зовём `rhs`, а не `b` — ниже `for (a, b), d` переписал бы переменную)
    M, rhs = assemble_mixed_coulomb_system(
        mesh, vs, ss, nu=NU, J_fn=lambda x: manufactured_periodic_rhs(x, NU)
    )

    # сопоставление x=0 → x=1 (геометрически, по всем x-вершинам)
    vmap = match_periodic_vertices(
        mesh, lambda c: _f(c, 0, 0.0), lambda c: _f(c, 0, 1.0),
        lambda c: np.array([c[0] + 1.0, c[1], c[2]]), tol=TOL,
    )

    yz_faces = [(1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)]

    def on_yz(v):
        return any(_f(V[v], ax, val) for ax, val in yz_faces)

    def edge_in_yz(a, b):
        return any(_f(V[a], ax, val) and _f(V[b], ax, val) for ax, val in yz_faces)

    def on_x(v):
        return _f(V[v], 0, 0.0) or _f(V[v], 0, 1.0)

    def edge_in_x(a, b):
        return (_f(V[a], 0, 0.0) and _f(V[b], 0, 0.0)) or (_f(V[a], 0, 1.0) and _f(V[b], 0, 1.0))

    # КОНФЛИКТНЫЕ DOF (на пересечении x-грани и yz-грани) → исключить из периодики (Dirichlet)
    excluded: set[int] = set()
    for v in range(mesh.n_vertices):
        if on_x(v) and on_yz(v):
            excluded.add(n_a + v)
    for (a, b), d in e2d.items():
        if edge_in_x(a, b) and edge_in_yz(a, b):
            excluded.add(int(d))

    red = build_periodic_reduction(vs, ss, vmap, antiperiodic=False, excluded=excluded)
    M_red, b_red = reduce_system(M, rhs, red)

    # Dirichlet на гранях y,z (рёбра в yz-гранях + вершины на yz) — в РЕДУЦИРОВАННОЙ системе
    dirichlet_global: set[int] = set()
    for v in range(mesh.n_vertices):
        if on_yz(v):
            dirichlet_global.add(n_a + v)
    for (a, b), d in e2d.items():
        if edge_in_yz(a, b):
            dirichlet_global.add(int(d))

    for g in dirichlet_global:
        r = int(red.reduced_of[g])
        assert r >= 0, "Dirichlet DOF оказался ведомым (ошибка классификации)"
        M_red[r, :] = 0.0
        M_red[:, r] = 0.0
        M_red[r, r] = 1.0
        b_red[r] = 0.0

    try:
        x_red = np.linalg.solve(M_red, b_red)
    except np.linalg.LinAlgError:
        x_red, *_ = np.linalg.lstsq(M_red, b_red, rcond=None)

    x = expand_solution(x_red, red)
    a = x[:n_a]
    err_a = l2_error_at_cell_centroids(vs, a, manufactured_periodic_A)
    err_curl = l2_curl_error_at_cell_centroids(vs, a, manufactured_periodic_curl)
    return err_a, err_curl


def test_periodic_bc_recovers_manufactured_with_order_one() -> None:
    e2a, e2c = _solve_periodic_mms(2)
    e4a, e4c = _solve_periodic_mms(4)
    e6a, e6c = _solve_periodic_mms(6)

    # восстановление точного периодического решения: монотонное убывание при измельчении
    assert e4a < e2a and e6a < e4a, (e2a, e4a, e6a)
    assert e4c < e2c and e6c < e4c, (e2c, e4c, e6c)
    # малая абсолютная ошибка на тонкой сетке (восстановили именно A_exact; cos(2πx) —
    # высокочастотное поле, потому пороги мягче, чем у гладких MMS)
    assert e6a < 4e-2, e6a
    assert e6c < 2.5e-1, e6c
    # порядок ~1 (Неделек 1-го рода): измельчение 4→6, отношение h = 1.5
    rate_a = math.log(e4a / e6a) / math.log(6.0 / 4.0)
    rate_c = math.log(e4c / e6c) / math.log(6.0 / 4.0)
    assert 0.6 < rate_a < 1.6, rate_a
    assert 0.6 < rate_c < 1.6, rate_c
