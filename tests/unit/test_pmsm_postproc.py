import numpy as np
import pytest

# P6: статический пост-процессинг — момент (Arkkio) + ЭДС-постоянная. Оракулы: cogging (I=0)
# пренебрежимо мал; момент реакции якоря антисимметричен по реверсу тока T(γ+π)≈−T(γ);
# нагруженный момент существенен; ЭДС-постоянная положительна и падает при демаге (через λ).

IPK, NPS, MESH = 20.0, 40.0, 0.004


@pytest.fixture(scope="module")
def machine():
    pytest.importorskip("gmsh")
    from magcore.domain.magnet_model import n42sh_magnet
    from magcore.domain.steel_curves import m270_35a_bh_curve
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        star_of_slots_layout,
    )
    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=MESH))
    return (g, n42sh_magnet(easy_axis=(1, 0, 0)), m270_35a_bh_curve(),
            star_of_slots_layout(g.params.n_slots, g.params.n_poles))


@pytest.fixture(scope="module")
def solves(machine):
    from magcore.fem2d.machines import solve_machine_static
    g, magnet, steel, layout = machine

    def _solve(**kw):
        return solve_machine_static(g, magnet, steel, T=20.0, max_iter=400, **kw)

    noload = _solve()                                                     # магнит, ХХ
    kw = dict(layout=layout, i_peak=IPK, turns_per_slot=NPS)
    load_p = _solve(gamma_elec=np.deg2rad(45.0), **kw)                    # q-осевой ток
    load_m = _solve(gamma_elec=np.deg2rad(225.0), **kw)                   # реверс (+180°)
    return noload, load_p, load_m


def test_cogging_negligible(machine, solves):
    from magcore.fem2d.machines.postproc import airgap_torque_arkkio
    g, *_ = machine
    noload, load_p, load_m = solves
    cog = airgap_torque_arkkio(g, noload.B_cells)
    loaded = max(abs(airgap_torque_arkkio(g, load_p.B_cells)),
                 abs(airgap_torque_arkkio(g, load_m.B_cells)))
    assert abs(cog) < 0.02 * loaded                                      # cogging << нагруженного


def test_torque_current_reversal_antisymmetric(machine, solves):
    from magcore.fem2d.machines.postproc import airgap_torque_arkkio
    g, *_ = machine
    _, load_p, load_m = solves
    tp = airgap_torque_arkkio(g, load_p.B_cells)
    tm = airgap_torque_arkkio(g, load_m.B_cells)
    assert tp * tm < 0.0                                                 # знаки противоположны
    assert abs(tp + tm) < 0.15 * abs(tp)                                # T(γ+π) ≈ −T(γ)
    assert abs(tp) > 0.5                                                 # существенный момент [Н·м]


def test_back_emf_constant_positive_and_scales_with_flux(machine, solves):
    from magcore.fem2d.machines import phase_flux_linkage
    from magcore.fem2d.machines.postproc import back_emf_constant, flux_linkage_amplitude
    g, _, _, layout = machine
    noload, *_ = solves
    lam = phase_flux_linkage(g, layout, noload.a, turns_per_slot=NPS)
    ke = back_emf_constant(g, lam)
    p = g.params.n_poles // 2
    assert ke > 0.0
    assert np.isclose(ke, p * flux_linkage_amplitude(lam))               # K_e = p·λ_m


def test_torque_scales_with_axial_length(machine, solves):
    # Момент ∝ осевая длина (2D-масштаб) — линейно.
    from magcore.fem2d.machines.postproc import airgap_torque_arkkio
    g, *_ = machine
    _, load_p, _ = solves
    t1 = airgap_torque_arkkio(g, load_p.B_cells, axial_length=0.030)
    t2 = airgap_torque_arkkio(g, load_p.B_cells, axial_length=0.060)
    assert np.isclose(t2, 2.0 * t1)
