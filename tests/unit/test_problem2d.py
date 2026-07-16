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
    pytest.importorskip("gmsh")
    from magcore.domain.steel_curves import m270_35a_bh_curve
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        pmsm_to_problem,
        solve_machine_static,
        star_of_slots_layout,
    )
    from magcore.fem2d.machines.excitation import winding_current_density

    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.004))
    magnet = n42sh_magnet((1, 0, 0))
    steel = m270_35a_bh_curve()

    # S1 (магнит-only) — общий путь = машинный путь поячеечно (тот же решатель Picard: проверка
    # эквивалентности СБОРКИ Problem2D↔машина, независимо от выбора метода).
    rm = solve_machine_static(g, magnet, steel, T=20.0, relaxation=0.1, max_iter=300)
    sp = solve_problem2d(pmsm_to_problem(g, magnet, steel, T=20.0),
                         method="picard", relaxation=0.1, max_iter=300)
    assert sp.converged
    assert np.allclose(sp.B_cells, rm.B_cells, atol=1e-8)

    # С током и нагревом (S2): тот же результат + та же карта демага.
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    jz = winding_current_density(g, lay, i_peak=30.0, gamma_elec=np.pi, turns_per_slot=40.0)
    rm2 = solve_machine_static(g, magnet, steel, T=140.0, layout=lay, i_peak=30.0,
                               gamma_elec=np.pi, turns_per_slot=40.0, relaxation=0.1, max_iter=300)
    sp2 = solve_problem2d(pmsm_to_problem(g, magnet, steel, T=140.0, j_cells=jz),
                          method="picard", relaxation=0.1, max_iter=300)
    assert np.allclose(sp2.B_cells, rm2.B_cells, atol=1e-8)
    assert sp2.risk.n_demagnetized == rm2.risk.n_demagnetized

    # Ньютон (дефолт) сходится к ТОМУ ЖЕ физическому решению (эталон Picard) — робастно.
    spn = solve_problem2d(pmsm_to_problem(g, magnet, steel, T=140.0, j_cells=jz), max_iter=60)
    assert spn.converged
    assert np.allclose(spn.B_cells, rm2.B_cells, atol=2e-3)
    assert spn.risk.n_demagnetized == rm2.risk.n_demagnetized
