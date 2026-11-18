import numpy as np
import pytest

from magcore.fem2d.machines.catalog import dp25_brushed_dc, outrunner_pmsm_uav
from magcore.fem2d.machines.mesh_policy import (
    MESH_AIR_GAP_PER_GAP,
    MESH_MAGNET,
    critical_mesh,
    with_critical_mesh,
)
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, Region
from magcore.fem2d.verification import FAIL, PASS, check_airgap_resolution

# П. 12 плана 2026-11-18: политика «магнит и зазор — всегда мельчайшая сетка» жила только в скрипте статьи;
# боевой каталог строил представительный 12/14 равномерной сеткой 3 мм — зазор 1 мм там не разрешён (проверка
# `check_airgap_resolution` сама это ловит). Теперь политика — в пакете и в умолчаниях каталога.


def test_critical_mesh_follows_the_gap_and_the_convergence_level():
    p = OutrunnerPMSMParams(n_slots=12, n_poles=14, air_gap=0.6e-3, mesh_size=3.0e-3)
    assert critical_mesh(p) == {"magnet": MESH_MAGNET, "air_gap": pytest.approx(0.6e-3 / MESH_AIR_GAP_PER_GAP)}
    q = with_critical_mesh(p)
    assert q.mesh_size_by_region == {"magnet": MESH_MAGNET, "air_gap": pytest.approx(0.2e-3)}
    assert q.mesh_size == p.mesh_size                       # прочие регионы — как были
    # уже заданное более мелкое помельчение сохраняется, чужие регионы не трогаются
    finer = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=3.0e-3,
                                mesh_size_by_region={"magnet": 0.2e-3, "tooth": 1.0e-3})
    r = with_critical_mesh(finer).mesh_size_by_region
    assert r["magnet"] == 0.2e-3 and r["tooth"] == 1.0e-3 and r["air_gap"] == pytest.approx(1.0e-3 / 3.0)


def test_catalog_default_resolves_the_air_gap_and_the_opt_out_does_not():
    # Оракул — независимая проверка разрешения зазора (≥3 элемента поперёк): у каталога по умолчанию она
    # проходит, у равномерной 3 мм (прежнее умолчание) — нет.
    for critical, expect in ((True, PASS), (False, FAIL)):
        d = outrunner_pmsm_uav(critical_mesh=critical)
        g = d.build_geometry()
        chk = check_airgap_resolution(g.mesh, g.mask(Region.AIR_GAP), d.params.air_gap)
        assert chk.status == expect, (critical, chk.value)
    assert outrunner_pmsm_uav().params.mesh_size_by_region["magnet"] == MESH_MAGNET


def test_dp25_already_finer_than_the_policy_is_unchanged():
    p = dp25_brushed_dc().params
    assert with_critical_mesh(p).mesh_size_by_region == p.mesh_size_by_region
    assert np.all(np.array([p.mesh_size_by_region[k] for k in ("magnet", "air_gap")])
                  <= np.array([MESH_MAGNET, p.air_gap / MESH_AIR_GAP_PER_GAP]))
