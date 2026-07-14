import numpy as np
import pytest

# P5: количественная необратимая потеря. Оракулы: холодный без потери (drop≈0); горячий —
# потеря есть, падение λ (ЭДС/момента) того же порядка, что средняя потеря Br, и монотонно
# растёт с T; агрегаты упорядочены (max≥mean, 0≤доля≤1); λ ХХ сбалансирована.

IPK, GAMMA_D, NPS = 30.0, np.pi, 40.0
MESH = 0.004


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
def impact_cold(machine):
    from magcore.fem2d.machines import evaluate_demag_impact
    g, magnet, steel, layout = machine
    return evaluate_demag_impact(g, magnet, steel, layout, i_peak=IPK,
                                 gamma_elec=GAMMA_D, turns_per_slot=NPS, T=20.0)


@pytest.fixture(scope="module")
def impact_hot(machine):
    from magcore.fem2d.machines import evaluate_demag_impact
    g, magnet, steel, layout = machine
    return evaluate_demag_impact(g, magnet, steel, layout, i_peak=IPK,
                                 gamma_elec=GAMMA_D, turns_per_slot=NPS, T=140.0)


def test_cold_no_irreversible_loss(impact_cold):
    assert impact_cold.load_converged
    assert impact_cold.aggregate.demag_area_fraction == 0.0
    assert impact_cold.aggregate.mean_loss_frac == 0.0
    assert abs(impact_cold.flux_linkage_drop_frac) < 1e-6      # λ не падает без демага


def test_hot_quantitative_loss(impact_hot):
    a = impact_hot.aggregate
    assert impact_hot.load_converged
    assert a.demag_area_fraction > 0.0                         # часть магнита за коленом
    assert a.mean_loss_frac > 0.0
    assert impact_hot.flux_linkage_drop_frac > 0.0             # машина реально теряет ЭДС/момент


def test_flux_drop_same_order_as_mean_loss(impact_hot):
    # Падение λ и средняя потеря Br — одного порядка (оба несколько процентов), падение λ
    # не абсурдно велико относительно потери Br (иначе — ошибка модели/latching).
    a = impact_hot.aggregate
    drop = impact_hot.flux_linkage_drop_frac
    assert 0.0 < drop < 0.10
    assert drop < 25.0 * a.mean_loss_frac                      # согласованность порядков


def test_loss_monotonic_in_temperature(impact_cold, impact_hot):
    assert impact_hot.flux_linkage_drop_frac > impact_cold.flux_linkage_drop_frac
    assert impact_hot.aggregate.demag_area_fraction > impact_cold.aggregate.demag_area_fraction


def test_aggregate_ordering(impact_hot):
    a = impact_hot.aggregate
    assert a.max_loss_frac >= a.mean_loss_frac >= 0.0
    assert 0.0 <= a.demag_area_fraction <= 1.0
    assert a.total_lost_br_area > 0.0


def test_noload_flux_linkage_balanced(impact_cold):
    # λ ХХ сбалансированной обмотки: сумма фаз ≈ 0, ненулевая амплитуда.
    lam = impact_cold.lam_nominal
    assert abs(float(lam.sum())) < 0.2 * np.abs(lam).max()
    assert np.abs(lam).max() > 0.0
