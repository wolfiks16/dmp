import numpy as np
import pytest

# Рабочая точка магнита ПО ОБЪЁМУ. Оракулы: H_op согласован с risk-map; объём=area×L;
# B_op>0 и P_c>0 на ХХ; за коленом 0% объёма на холоде, >0 под горячей нагрузкой;
# перцентили монотонны; и главное — «толще магнит → выше рабочая точка» (моно по объёму).

IPK, NPS, MESH = 30.0, 40.0, 0.004


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
def op_noload(machine):
    from magcore.fem2d.machines import magnet_operating_point, solve_machine_static
    g, magnet, steel, _ = machine
    r = solve_machine_static(g, magnet, steel, T=20.0, max_iter=400)
    return r, magnet_operating_point(g, r, magnet, T=20.0)


@pytest.fixture(scope="module")
def op_hot(machine):
    from magcore.fem2d.machines import magnet_operating_point, solve_machine_static
    g, magnet, steel, layout = machine
    r = solve_machine_static(g, magnet, steel, T=140.0, layout=layout,
                             i_peak=IPK, gamma_elec=np.pi, turns_per_slot=NPS, max_iter=400)
    return r, magnet_operating_point(g, r, magnet, T=140.0)


def test_H_op_matches_risk_map(op_noload):
    r, op = op_noload
    # Та же рабочая точка, что в risk-map (два независимых извлечения из H_cells·e/μ0).
    assert np.allclose(op.H_op, r.risk.H_par)
    assert np.array_equal(op.cell_indices, r.risk.cell_indices)


def test_volume_equals_area_times_length(machine, op_noload):
    from magcore.fem2d.machines import Region
    g, *_ = machine
    _, op = op_noload
    areas = sum(g.mesh.cell_area(int(c)) for c in np.where(g.region == int(Region.MAGNET))[0])
    assert np.isclose(op.total_volume, areas * g.params.axial_length, rtol=1e-9)


def test_noload_operating_point_physical(op_noload):
    _, op = op_noload
    assert float(np.average(op.B_op, weights=op.cell_volume)) > 0.0   # индукция вдоль оси >0
    assert np.all(op.permeance > 0.0)                                  # коэфф. проницаемости >0
    assert op.worst_H_op() < 0.0                                       # рабочая точка во 2-м квадранте
    assert op.volume_fraction_below(op.knee_field) == 0.0             # ничего за коленом на ХХ 20°


def test_hot_load_volume_past_knee(op_hot):
    _, op = op_hot
    frac = op.volume_fraction_below(op.knee_field)
    assert frac > 0.0                                                  # часть ОБЪЁМА за коленом
    assert 0.0 < frac < 1.0


def test_percentiles_monotone_and_bracket_mean(op_noload):
    _, op = op_noload
    qs = np.linspace(0.0, 1.0, 11)
    pv = op.percentiles_H_op(qs)
    assert np.all(np.diff(pv) >= -1e-6)                                # неубывающие
    mean = op.volume_weighted_mean_H_op()
    assert op.worst_H_op() <= mean <= float(op.H_op.max())


def test_higher_operating_point_with_thicker_magnet(machine):
    # Твоя физика как оракул: толще магнит → выше P_c → выше рабочая точка (H_op ближе к 0).
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        magnet_operating_point,
        solve_machine_static,
    )
    _, magnet, steel, _ = machine
    ops = []
    for h_mag in (0.0020, 0.0050):
        g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(h_magnet=h_mag, mesh_size=MESH))
        r = solve_machine_static(g, magnet, steel, T=20.0, max_iter=400)
        ops.append(magnet_operating_point(g, r, magnet, T=20.0))
    thin, thick = ops
    # Толстый магнит: рабочая точка выше (mean H_op менее отрицателен) и P_c больше.
    assert thick.volume_weighted_mean_H_op() > thin.volume_weighted_mean_H_op()
    assert thick.volume_weighted_mean_permeance() > thin.volume_weighted_mean_permeance()
