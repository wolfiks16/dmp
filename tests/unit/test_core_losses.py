import math

import numpy as np

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.magnet_loss import magnet_segment_width
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.scenario import machine_scenario
from magcore.fem2d.machines.thermal_scenario import (
    precompute_core_loss_density,
    run_machine_thermal_demag,
)

# P-B5: интеграция источников (сталь статора + ротор-сторона) как extra_loss. Оракулы:
#  карта потерь ядра лежит в правильных регионах (железо+магнит, не воздух/паз);
#  ПЕЙЛОАД B-full — магнит греется от СОБСТВЕННОГО источника (T выше среды), чего не было,
#  пока магнит зависел только от кондукции меди через изолирующий зазор.

MESH = 0.005


def _params():
    return OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH)


def test_core_loss_density_in_correct_regions():
    p = _params()
    magnet, steel = n42sh_magnet((1.0, 0.0, 0.0)), m270_35a_bh_curve()
    q = precompute_core_loss_density(
        p, magnet, steel, speed_rpm=7000.0, i_peak=40.0, gamma_elec=0.0, turns_per_slot=20.0,
        T=60.0, sigma_pm=0.79e6, magnet_seg_width=magnet_segment_width(p, 1),
        mech_span=2.0 * math.pi / p.n_slots, n_positions=6, relaxation=0.1, max_iter=200,
    )
    geo = build_outrunner_spm_pmsm(p)
    assert q.shape == (geo.mesh.n_cells,)
    # потери есть в железе статора, магните и ярме ротора
    for reg in (Region.STATOR_YOKE, Region.TOOTH, Region.MAGNET, Region.ROTOR_YOKE):
        assert np.any(q[geo.region == int(reg)] > 0.0), f"нет потерь в {reg}"
    # и НЕТ в воздухе зазора и пазах (медь считается отдельно внутри связки)
    assert np.allclose(q[geo.region == int(Region.AIR_GAP)], 0.0)
    assert np.allclose(q[geo.region == int(Region.SLOT)], 0.0)


def test_core_losses_heat_the_magnet():
    # ПЕЙЛОАД ВСЕЙ B-full: с потерями ядра магнит получает СВОЙ источник тепла и греется
    # выше среды; без них (только медь + изолирующий зазор) магнит остаётся у ambient.
    p = _params()
    sc = machine_scenario(p, n42sh_magnet((1.0, 0.0, 0.0)), m270_35a_bh_curve())
    common = dict(i_peak=40.0, turns_per_slot=20.0, gamma_elec=0.0, slot_fill=0.45,
                  h=20.0, h_in=800.0, T_frame=60.0, T_amb=60.0, T0=60.0,
                  dt=2.0, n_steps=6, max_substeps=8)
    no_core = run_machine_thermal_demag(sc, **common)
    with_core = run_machine_thermal_demag(
        sc, core_losses=True, speed_rpm=7000.0, sigma_pm=0.79e6,
        magnet_seg_width=magnet_segment_width(p, 1),
        loss_mech_span=2.0 * math.pi / p.n_slots, loss_n_positions=6, **common,
    )
    # ядро добавляет потери (сталь+магнит к меди) ⇒ суммарное тепловыделение выше
    assert with_core.loss_power_final > no_core.loss_power_final * 1.05
    # и магнит получает СВОЙ источник ⇒ греется сильнее, чем от одной кондукции меди
    assert with_core.T_magnet_max > no_core.T_magnet_max + 0.1
