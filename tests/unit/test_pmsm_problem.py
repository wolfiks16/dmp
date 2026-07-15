import dataclasses

import numpy as np
import pytest

from magcore.fem2d.machines.winding import star_of_slots_layout

# P7: согласованная модель задачи + один вызов «модель → полный расчёт». Оракулы: валидная
# постановка проходит; некорректные (несогласованные размеры, T вне диапазона, S1 с T≠20,
# ток без витков, плохие параметры решателя) отвергаются; сквозной прогон S1 и S2.


@pytest.fixture(scope="module")
def machine():
    pytest.importorskip("gmsh")
    from magcore.domain.magnet_model import n42sh_magnet
    from magcore.domain.steel_curves import m270_35a_bh_curve
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
    )
    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.004))
    return (g, n42sh_magnet(easy_axis=(1, 0, 0)), m270_35a_bh_curve(),
            star_of_slots_layout(g.params.n_slots, g.params.n_poles))


@pytest.fixture(scope="module")
def base_problem(machine):
    from magcore.fem2d.machines import MachineProblem, Scenario
    g, magnet, steel, layout = machine
    return MachineProblem(geometry=g, magnet=magnet, steel=steel, layout=layout,
                          scenario=Scenario.S1, T=20.0)


def test_valid_problem_passes(base_problem):
    assert base_problem.validate() == []
    base_problem.check()                                   # не бросает


def test_rejects_size_mismatch(base_problem):
    bad = dataclasses.replace(base_problem, layout=star_of_slots_layout(6, 4))
    assert any("размер" in s.lower() or "n_slots" in s for s in bad.validate())
    with pytest.raises(ValueError):
        bad.check()


def test_rejects_s1_with_nonroom_temperature(base_problem):
    bad = dataclasses.replace(base_problem, T=100.0)        # S1 требует 20 °C
    assert bad.validate()
    with pytest.raises(ValueError):
        bad.check()


def test_rejects_overheated_temperature(base_problem):
    from magcore.fem2d.machines import Scenario
    limit = base_problem.magnet.temperature_limit()
    bad = dataclasses.replace(base_problem, scenario=Scenario.S2, T=limit + 50.0)
    assert bad.validate()
    with pytest.raises(ValueError):
        bad.check()


def test_rejects_current_without_turns(base_problem):
    bad = dataclasses.replace(base_problem, i_peak=30.0, turns_per_slot=0.0)
    assert bad.validate()


def test_rejects_bad_solver_params(base_problem):
    assert dataclasses.replace(base_problem, relaxation=2.0).validate()
    assert dataclasses.replace(base_problem, max_iter=0).validate()
    assert dataclasses.replace(base_problem, tol=0.0).validate()


def test_s3_allows_prescribed_temperature(base_problem):
    from magcore.fem2d.machines import Scenario
    ok = dataclasses.replace(base_problem, scenario=Scenario.S2, T=120.0)
    assert ok.validate() == []


def test_solve_s1_end_to_end(base_problem):
    from magcore.fem2d.machines import solve_machine_problem
    sol = solve_machine_problem(base_problem)              # S1, магнит-only
    assert sol.field.converged
    assert sol.operating_point.total_volume > 0.0
    assert abs(sol.torque) < 1e-2                          # cogging мал
    assert sol.loss.demag_area_fraction == 0.0            # 20 °C без демага
    assert sol.impact is None


def test_solve_s3_with_current_and_impact(machine):
    from magcore.fem2d.machines import MachineProblem, Scenario, solve_machine_problem
    g, magnet, steel, layout = machine
    prob = MachineProblem(geometry=g, magnet=magnet, steel=steel, layout=layout,
                          scenario=Scenario.S2, T=140.0, i_peak=30.0,
                          gamma_elec=np.pi, turns_per_slot=40.0)
    sol = solve_machine_problem(prob, assess_impact=True)
    assert sol.field.converged
    assert sol.impact is not None
    assert sol.impact.flux_linkage_drop_frac > 0.0        # необратимая потеря есть
    assert sol.operating_point.volume_fraction_below(sol.operating_point.knee_field) > 0.0
    assert sol.back_emf_constant > 0.0


def test_solve_rejects_invalid(base_problem):
    from magcore.fem2d.machines import solve_machine_problem
    bad = dataclasses.replace(base_problem, T=500.0)
    with pytest.raises(ValueError):
        solve_machine_problem(bad)
