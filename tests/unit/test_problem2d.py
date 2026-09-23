import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.model import (
    Air,
    MagnetMaterial,
    Problem2D,
    Region2D,
    solve_problem2d,
)

# Общая регион-объектная модель задачи. Оракулы: (1) валидация постановки; (2) общий
# линейный кейс (НЕ машина) решается; (3) ГЛАВНЫЙ — Problem2D воспроизводит машинный
# решатель ⇒ PMSM демонтирован из «зашитой модели» в один из поставщиков общей задачи.


def _air_problem():
    mesh = build_structured_rectangle_tri_mesh(3, 3)
    nc = mesh.n_cells
    return Problem2D(mesh=mesh, cell_region=np.zeros(nc, int),
                     regions={0: Region2D(0, "air", Air())})


def test_validate_ok():
    assert _air_problem().validate() == []


def test_validate_missing_material():
    p = _air_problem()
    bad = Problem2D(mesh=p.mesh, cell_region=np.ones(p.mesh.n_cells, int), regions=p.regions)
    assert bad.validate()
    with pytest.raises(ValueError):
        bad.check()


def test_validate_magnet_needs_axis():
    mesh = build_structured_rectangle_tri_mesh(2, 2)
    p = Problem2D(mesh=mesh, cell_region=np.zeros(mesh.n_cells, int),
                  regions={0: Region2D(0, "mag", MagnetMaterial(n42sh_magnet((1, 0, 0))))})
    assert any("magnet_axis" in s for s in p.validate())


def test_validate_jcells_shape():
    p = _air_problem()
    bad = Problem2D(mesh=p.mesh, cell_region=p.cell_region, regions=p.regions,
                    j_cells=np.zeros(p.mesh.n_cells + 1))
    assert bad.validate()


def test_linear_air_current_solves():
    # Общий кейс без магнита/стали: воздух + ток → сходится, поле ненулевое, risk отсутствует.
    mesh = build_structured_rectangle_tri_mesh(8, 8)
    nc = mesh.n_cells
    cen = np.array([mesh.cell_centroid(c) for c in range(nc)])
    j = np.zeros(nc)
    j[(np.abs(cen[:, 0] - 0.5) < 0.18) & (np.abs(cen[:, 1] - 0.5) < 0.18)] = 1.0e5
    p = Problem2D(mesh=mesh, cell_region=np.zeros(nc, int),
                  regions={0: Region2D(0, "air", Air())}, j_cells=j)
    sol = solve_problem2d(p, max_iter=40)
    assert sol.converged
    assert np.max(np.hypot(sol.B_cells[:, 0], sol.B_cells[:, 1])) > 0.0
    assert sol.risk is None


def test_problem2d_matches_machine_solver():
    """
    PMSM — лишь ПОСТАВЩИК общей задачи. (1) Машинная сборка ν (ею пользуются расчёты с замороженным
    источником магнита и ядро К6′) совпадает со сборкой Problem2D через мост бит в бит — в том числе
    в насыщении. (2) Машинный статический расчёт и есть общий расчёт на задаче из моста: у физики
    одна реализация (Л-107). Совпадение с прежней схемой — в test_machine_magnet_knee.py.
    """
    pytest.importorskip("gmsh")
    from magcore.domain.steel_curves import m270_35a_bh_curve
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        machine_reluctivity,
        pmsm_to_problem,
        solve_machine_static,
        star_of_slots_layout,
    )
    from magcore.fem2d.machines.excitation import winding_current_density
    from magcore.fem2d.model.problem import _reluctivity

    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.004))
    magnet = n42sh_magnet((1, 0, 0))
    steel = m270_35a_bh_curve()

    # (1) Сборка: ν(B) при произвольном поле до ~3 Тл (сталь глубоко в насыщении) — тождественно.
    nu_m, nu0_m, mask_m, _ = machine_reluctivity(g, magnet, steel)
    prob = pmsm_to_problem(g, magnet, steel, T=20.0)
    nu_p, nu0_p = _reluctivity(prob)
    B = np.random.default_rng(7).normal(scale=1.0, size=(g.mesh.n_cells, 2))
    assert np.array_equal(nu_m(B), nu_p(B))
    assert np.array_equal(nu0_m, nu0_p)
    assert np.array_equal(mask_m, prob.magnet_mask())

    # (2) С током и нагревом (S2, за коленом): машинный расчёт = общий на задаче из моста.
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    jz = winding_current_density(g, lay, i_peak=30.0, gamma_elec=np.pi, turns_per_slot=40.0)
    rm = solve_machine_static(g, magnet, steel, T=140.0, layout=lay, i_peak=30.0,
                              gamma_elec=np.pi, turns_per_slot=40.0, max_iter=60)
    sp = solve_problem2d(pmsm_to_problem(g, magnet, steel, T=140.0, j_cells=jz), max_iter=60)
    assert rm.converged and sp.converged
    assert np.array_equal(sp.B_cells, rm.B_cells)
    assert np.array_equal(sp.risk.H_par, rm.risk.H_par)
    assert rm.risk.n_demagnetized > 0
