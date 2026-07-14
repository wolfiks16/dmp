import math

import numpy as np
import pytest

pytest.importorskip("gmsh")  # генератор геометрии требует gmsh (решатель — нет)

from magcore.fem2d.machines import (  # noqa: E402
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)


def _geo():
    # Покрупнее для скорости теста.
    p = OutrunnerPMSMParams(mesh_size=0.0018)
    return build_outrunner_spm_pmsm(p), p


def test_mesh_valid_and_tiles_annulus():
    g, p = _geo()
    assert g.mesh.n_cells > 0
    total = sum(g.region_areas().values())
    annulus = math.pi * (p.R_out ** 2 - p.R_bore ** 2)
    assert total == pytest.approx(annulus, rel=1e-3)   # области точно тайлят кольцо


def test_region_areas_match_analytic():
    g, p = _geo()
    a = g.region_areas()
    pi = math.pi
    A_band_s = pi * (p.R_s_out ** 2 - p.R_sy ** 2)
    A_band_m = pi * (p.R_mag_out ** 2 - p.R_mag_in ** 2)
    assert a["stator_yoke"] == pytest.approx(pi * (p.R_sy ** 2 - p.R_bore ** 2), rel=2e-2)
    assert a["tooth"] == pytest.approx(p.tooth_width_frac * A_band_s, rel=2e-2)
    assert a["slot"] == pytest.approx((1 - p.tooth_width_frac) * A_band_s, rel=2e-2)
    assert a["magnet"] == pytest.approx(p.magnet_embrace * A_band_m, rel=2e-2)
    assert a["rotor_yoke"] == pytest.approx(pi * (p.R_out ** 2 - p.R_mag_out ** 2), rel=2e-2)


def test_magnet_axis_is_radial_and_alternates():
    g, p = _geo()
    mag = np.where(g.mask(Region.MAGNET))[0]
    assert mag.size > 0
    axes = g.magnet_easy_axis[mag]
    cent = g.mesh.vertices[g.mesh.cells[mag]].mean(axis=1)

    # ось единичная и коллинеарна радиусу (|ось × радиус_орт| ≈ 0).
    norms = np.linalg.norm(axes, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    rhat = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    cross = axes[:, 0] * rhat[:, 1] - axes[:, 1] * rhat[:, 0]
    assert np.max(np.abs(cross)) < 1e-9

    # полярность чередуется: присутствуют оба знака радиальной проекции, примерно поровну.
    radial_sign = np.sign(np.einsum("ij,ij->i", axes, rhat))
    assert np.any(radial_sign > 0) and np.any(radial_sign < 0)
    assert abs(radial_sign.sum()) < 0.25 * radial_sign.size


def test_all_regions_present():
    g, _ = _geo()
    present = set(np.unique(g.region))
    for r in Region:
        assert int(r) in present, f"регион {r.name} отсутствует"
