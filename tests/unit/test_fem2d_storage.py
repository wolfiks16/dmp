import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.model import Air, GeoObject, MagnetMaterial, SteelMaterial, auto_domain, build_object_problem
from magcore.fem2d.model.object_geometry import object_problem_from_mesh
from magcore.fem2d.model.problem import solve_problem2d
from magcore.fem2d.model.storage import (
    FIELD2D_FORMAT,
    STALE_FACTOR,
    SOLVER_TOL,
    field_payload2d,
    relative_residual,
    restore_saved_field2d,
    saved_array,
    saved_mesh2d,
)

# ПОЛЕ 2D В ФАЙЛЕ РАСЧЁТА И ЕГО ПРОВЕРКА (как этап 3D-9 для 3D; решение Sergey 2026-09-25). Оракулы:
#  (1) та же задача, собранная заново на сетке из файла, при сохранённом A_z даёт ТУ ЖЕ невязку, что у
#      решателя, до бита: код и данные те же, итераций нет; поле и карта риска — те же;
#  (2) изменились уравнения — невязка больше допуска: другая марка магнита, другая температура, другой ток
#      обмотки (у машины ток при открытии считает текущий код по I, γ, виткам, а не берётся из файла);
#  (3) A в узле Дирихле другой — это решение другой задачи: невязка бесконечна;
#  (4) повреждённое сохранённое поле — понятная ошибка, а не молчаливая подмена.

MM = 1.0e-3
STEEL = SteelMaterial(m270_35a_bh_curve())


def _objects(magnet=None):
    steel = GeoObject("сталь", "rect", {"cx": -8 * MM, "cy": 0.0, "w": 10 * MM, "h": 20 * MM}, STEEL)
    mag = GeoObject("магнит", "circle", {"cx": 8 * MM, "cy": 0.0, "r": 5 * MM},
                    MagnetMaterial(magnet or n42sh_magnet((1, 0, 0))), magnet_dir=(1.0, 0.0), mesh_size=0.9 * MM)
    objs = [steel, mag]
    return objs, auto_domain(objs, material=Air(), margin_frac=0.6, mesh_size=2.5 * MM)


@pytest.fixture(scope="module")
def solved():
    pytest.importorskip("gmsh")
    objs, dom = _objects()
    sol = solve_problem2d(build_object_problem(objs, dom, default_mesh_size=2.5 * MM, T=20.0))
    assert sol.converged
    return sol


def _restore(payload, magnet=None, T=None):
    saved = saved_mesh2d(payload)
    objs, dom = _objects(magnet)
    prob = object_problem_from_mesh(objs, dom, saved.vertices, saved.cells, saved.cell_region,
                                    T=saved.T if T is None else T)
    return restore_saved_field2d(saved, prob)


def test_the_same_problem_on_the_saved_mesh_gives_the_solver_residual_to_the_bit(solved):
    payload = field_payload2d(solved)
    assert payload["format"] == FIELD2D_FORMAT and payload["residual"] == relative_residual(solved)
    r = _restore(payload)
    assert r.ok and r.residual == r.stored_residual == relative_residual(solved)
    assert r.residual <= SOLVER_TOL                                   # решение сошлось
    f, g = solved.field, r.solution.field
    assert np.array_equal(g.a, f.a) and np.array_equal(g.B_cells, f.B_cells) and np.array_equal(g.H_cells, f.H_cells)
    assert np.array_equal(r.solution.risk.loss, solved.risk.loss) and np.array_equal(r.solution.risk.H_par, solved.risk.H_par)
    assert r.solution.field.n_iterations == 0


def test_changed_equations_make_the_saved_field_stale(solved):
    payload = field_payload2d(solved)
    limit = max(SOLVER_TOL, STALE_FACTOR * payload["residual"])
    other_grade = _restore(payload, magnet=sm2co17_magnet((1, 0, 0)))
    hotter = _restore(payload, T=120.0)
    for r in (other_grade, hotter):
        assert not r.ok and r.residual > limit
        assert r.stored_residual == payload["residual"]


def test_a_different_value_at_a_dirichlet_node_is_another_problem(solved):
    payload = field_payload2d(solved)
    saved = saved_mesh2d(payload)
    a = saved.a.copy()
    edge = int(np.argmax(np.abs(saved.vertices[:, 0])))              # узел на внешней границе области
    a[edge] = 1.0e-6
    objs, dom = _objects()
    prob = object_problem_from_mesh(objs, dom, saved.vertices, saved.cells, saved.cell_region, T=saved.T)
    r = restore_saved_field2d(type(saved)(**{**saved.__dict__, "a": a}), prob)
    assert not r.ok and r.residual == float("inf")


def test_damaged_saved_field_is_a_clear_error(solved):
    good = field_payload2d(solved)
    for bad in ({**good, "format": "другое"}, {**good, "version": 99}, {k: v for k, v in good.items() if k != "a"},
                {**good, "a": good["vertices"]}, {**good, "T": "жарко"}, None):
        with pytest.raises(ValueError):
            saved_mesh2d(bad)
    with pytest.raises(ValueError):
        saved_array(good, "magnet_axis", "<f8", 2 * good["n_cells"])     # у свободной геометрии оси в файле нет
    saved = saved_mesh2d(good)
    objs, dom = _objects()
    with pytest.raises(ValueError):                                   # регион — номер объекта модели
        object_problem_from_mesh(objs, dom, saved.vertices, saved.cells, saved.cell_region + 5, T=20.0)


@pytest.fixture(scope="module")
def machine():
    pytest.importorskip("gmsh")
    from magcore.fem2d.machines import (
        MachineScenario,
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        star_of_slots_layout,
    )
    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.004))
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    scen = MachineScenario(geometry=g, magnet=n42sh_magnet((1, 0, 0)), steel=m270_35a_bh_curve(), layout=lay)
    sol = scen.solve(T=20.0, i_peak=20.0, gamma_elec=np.deg2rad(45.0), turns_per_slot=40.0)
    assert sol.converged
    return g, lay, sol


def _machine_problem(g, lay, saved, payload, *, i_peak):
    from magcore.fem2d.machines import MachineScenario
    from magcore.fem2d.machines.bridge import machine_geometry_on_mesh
    nc = saved.cells.shape[0]
    g2 = machine_geometry_on_mesh(saved.vertices, saved.cells, saved.cell_region,
                                  saved_array(payload, "magnet_axis", "<f8", 2 * nc).reshape(nc, 2),
                                  saved_array(payload, "slot_id", "<i4", nc), g.params)
    scen = MachineScenario(geometry=g2, magnet=n42sh_magnet((1, 0, 0)), steel=m270_35a_bh_curve(), layout=lay)
    return scen.to_problem(T=saved.T, i_peak=i_peak, gamma_elec=np.deg2rad(45.0), turns_per_slot=40.0)


def test_machine_field_is_checked_with_the_winding_current_of_the_current_code(machine):
    g, lay, sol = machine
    payload = field_payload2d(sol, magnet_axis=g.magnet_easy_axis, slot_id=g.slot_id)
    saved = saved_mesh2d(payload)
    same = restore_saved_field2d(saved, _machine_problem(g, lay, saved, payload, i_peak=20.0))
    assert same.ok and same.residual == relative_residual(sol)
    assert np.array_equal(same.solution.field.B_cells, sol.field.B_cells)
    # ток при открытии считает текущий код по I, γ и виткам: другой ток — другая задача
    other = restore_saved_field2d(saved, _machine_problem(g, lay, saved, payload, i_peak=10.0))
    assert not other.ok and other.residual > max(SOLVER_TOL, STALE_FACTOR * payload["residual"])
