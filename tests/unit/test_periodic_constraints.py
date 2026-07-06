from __future__ import annotations

import numpy as np
import pytest

from magcore.femcore.periodic import (
    build_periodic_reduction,
    expand_solution,
    match_periodic_vertices,
    reduce_system,
)
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh

TOL = 1e-9


def _x0(c):
    return abs(c[0] - 0.0) < TOL


def _x1(c):
    return abs(c[0] - 1.0) < TOL


def _shift_x(c):
    return np.array([c[0] + 1.0, c[1], c[2]])


def _setup(n=2):
    mesh = build_structured_unit_cube_tetra_mesh(n)
    return mesh, NedelecP1Space.from_mesh(mesh), LagrangeP1Space(mesh)


def test_vertex_matching_translation_box() -> None:
    mesh, _, _ = _setup(2)
    vmap = match_periodic_vertices(mesh, _x0, _x1, _shift_x, tol=TOL)
    assert len(vmap) == 9  # (n+1)^2 узлов на грани x=0 при n=2
    V = mesh.vertices
    for s, m in vmap.items():
        assert _x0(V[s]) and _x1(V[m])
        assert np.allclose(V[m], _shift_x(V[s]), atol=TOL)  # геометрия совпала
    assert len(set(vmap.values())) == len(vmap)  # биекция


def test_reduction_shapes_and_block_structure() -> None:
    mesh, vs, ss = _setup(2)
    vmap = match_periodic_vertices(mesh, _x0, _x1, _shift_x)
    red = build_periodic_reduction(vs, ss, vmap, antiperiodic=False)

    n_a, n_p = vs.ndofs, ss.ndofs
    assert red.n_full == n_a + n_p
    assert red.n_red == red.n_full - red.slave_dofs.size
    assert red.T.shape == (red.n_full, red.n_red)
    # ровно один slave на каждую slave-вершину p + slave-рёбра грани x=0
    n_slave_verts = 9
    assert (red.slave_dofs >= n_a).sum() == n_slave_verts  # p-slaves
    assert (red.slave_dofs < n_a).sum() >= 1               # есть рёбра-slave

    # БЛОЧНОСТЬ T по (a,p): нет перекрёстных a↔p связей (сохраняет седловую структуру).
    retained = [i for i in range(red.n_full) if i not in set(red.slave_dofs.tolist())]
    for col, g in enumerate(retained):
        nz = np.nonzero(red.T[:, col])[0]
        if g < n_a:  # a-столбец → только a-строки
            assert np.all(nz < n_a), "a-столбец T затрагивает p-DOF (нарушена блочность)"
        else:        # p-столбец → только p-строки
            assert np.all(nz >= n_a), "p-столбец T затрагивает a-DOF (нарушена блочность)"


def test_periodic_constant_field_reconstructed_exactly() -> None:
    """
    СЕМАНТИЧЕСКИЙ якорь знаков рёбер: константное поле A0 (циркуляции по рёбрам) +
    константный p ОБЯЗАНЫ быть точно в образе T (slave = orient·master). Любая ошибка
    в master/знаке ребра ⇒ T·v_red ≠ v. Для трансляции константное поле периодично.
    """
    mesh, vs, ss = _setup(2)
    vmap = match_periodic_vertices(mesh, _x0, _x1, _shift_x)
    red = build_periodic_reduction(vs, ss, vmap, antiperiodic=False)

    V = mesh.vertices
    A0 = np.array([0.3, 1.0, -0.7])  # произвольное постоянное поле
    e2d = vs.edge_to_dof_map()
    v = np.zeros(red.n_full)
    for (a, b), d in e2d.items():
        v[d] = A0 @ (V[b] - V[a])     # циркуляция low→high (Неделек DOF)
    v[vs.ndofs:] = 2.5                 # константный p (периодичен)

    retained = [i for i in range(red.n_full) if i not in set(red.slave_dofs.tolist())]
    v_red = v[retained]
    assert np.allclose(red.T @ v_red, v, atol=1e-12), "периодическое поле не в образе T"


def test_antiperiodic_flips_signs_vs_periodic() -> None:
    mesh, vs, ss = _setup(2)
    vmap = match_periodic_vertices(mesh, _x0, _x1, _shift_x)
    red_p = build_periodic_reduction(vs, ss, vmap, antiperiodic=False)
    red_a = build_periodic_reduction(vs, ss, vmap, antiperiodic=True)
    # тот же шаблон разреженности, противоположные коэффициенты на slave-строках.
    assert np.array_equal(red_p.slave_dofs, red_a.slave_dofs)
    for s in red_p.slave_dofs:
        rp, ra = red_p.T[s], red_a.T[s]
        assert np.allclose(ra, -rp), "анти-период не инвертировал знак slave-связи"


def test_reduction_preserves_symmetry() -> None:
    mesh, vs, ss = _setup(2)
    vmap = match_periodic_vertices(mesh, _x0, _x1, _shift_x)
    red = build_periodic_reduction(vs, ss, vmap)
    rng = np.random.default_rng(0)
    M = rng.standard_normal((red.n_full, red.n_full))
    M = M + M.T
    b = rng.standard_normal(red.n_full)
    M_red, b_red = reduce_system(M, b, red)
    assert M_red.shape == (red.n_red, red.n_red)
    assert np.allclose(M_red, M_red.T, atol=1e-10)  # TᵀMT симметрична
    # expand обратно совместим по форме
    assert expand_solution(np.ones(red.n_red), red).shape == (red.n_full,)


def test_match_rejects_nonconforming_transform() -> None:
    mesh, _, _ = _setup(2)
    # неверный сдвиг (нет совпадения master) ⇒ ValueError
    with pytest.raises(ValueError):
        match_periodic_vertices(mesh, _x0, _x1, lambda c: np.array([c[0] + 0.37, c[1], c[2]]))
