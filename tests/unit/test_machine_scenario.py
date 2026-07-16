import numpy as np
import pytest

# «Машина как сценарий» поверх общей Problem2D. Оракулы: момент сценария == машинный/общий
# Арккио; рабочая точка == risk-map; ЭДС-постоянная > 0; свип момента по углу тока (d/q)
# антисимметричен по реверсу (момент ∝ i_q). Всё через общую базу + общий пост-проц.

IPK, NPS, MESH = 20.0, 40.0, 0.004


@pytest.fixture(scope="module")
def scen():
    pytest.importorskip("gmsh")
    from magcore.domain.magnet_model import n42sh_magnet
    from magcore.domain.steel_curves import m270_35a_bh_curve
    from magcore.fem2d.machines import OutrunnerPMSMParams, machine_scenario
    return machine_scenario(OutrunnerPMSMParams(mesh_size=MESH),
                            n42sh_magnet((1, 0, 0)), m270_35a_bh_curve())


@pytest.fixture(scope="module")
def sol_load(scen):
    return scen.solve(T=20.0, i_peak=IPK, gamma_elec=np.deg2rad(45.0), turns_per_slot=NPS, max_iter=300)


def test_scenario_torque_matches_machine_and_general(scen, sol_load):
    from magcore.fem2d.machines.postproc import airgap_torque_arkkio
    from magcore.fem2d.model import torque_arkkio
    g = scen.geometry
    t = scen.torque(sol_load)
    t_mach = airgap_torque_arkkio(g, sol_load.B_cells, axial_length=g.params.axial_length)
    t_gen = torque_arkkio(sol_load, g.params.R_s_out, g.params.R_mag_in, axial_length=g.params.axial_length)
    assert np.isclose(t, t_mach, rtol=1e-9) and np.isclose(t, t_gen, rtol=1e-9)
    assert abs(t) > 0.1


def test_scenario_operating_point_matches_risk(scen, sol_load):
    op = scen.operating_point(sol_load)
    assert np.allclose(op.H_op, sol_load.risk.H_par)
    assert op.total_volume > 0.0


def test_scenario_back_emf_positive(scen):
    nl = scen.solve(T=20.0, max_iter=300)                       # холостой ход (ток=0)
    ke = scen.back_emf_constant(nl, turns_per_slot=NPS)
    assert ke > 0.0
    assert abs(scen.torque(nl)) < 1e-2                          # cogging при I=0 мал


def test_scenario_torque_vs_current_angle_antisymmetric(scen):
    t = scen.torque_vs_current_angle(np.deg2rad([45.0, 225.0]), i_peak=IPK, turns_per_slot=NPS,
                                     T=20.0, max_iter=300)
    assert t[0] * t[1] < 0.0                                    # реверс тока → смена знака момента
    assert abs(t[0] + t[1]) < 0.15 * abs(t[0])                  # T(γ+π) ≈ −T(γ)
    assert abs(t[0]) > 0.1
