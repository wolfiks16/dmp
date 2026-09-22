import math

import numpy as np

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.magnet_loss import (
    magnet_eddy_loss_density_from_waveform,
    magnet_rotor_loss_density,
    magnet_segment_width,
)
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)

# P-B4: вихревые в магните + сталь ротора (система ротора). Оракулы:
#  аналитика пластины P=σw²(2πf)²B²/24; масштаб f²; сегментация 1/N²; отношение σ;
#  реальный свип — q>0 только в магните+ярме ротора.


def _sinusoid(N, Bm, k=1, n_probes=1):
    theta = 2.0 * math.pi * np.arange(N) / N
    B = np.zeros((N, n_probes, 2))
    B[:, :, 0] = (Bm * np.sin(k * theta))[:, None]
    return B


def test_eddy_matches_slab_analytic():
    sigma, w, f, Bm, N = 0.8e6, 3.0e-3, 800.0, 0.05, 64
    q = magnet_eddy_loss_density_from_waveform(
        _sinusoid(N, Bm, n_probes=2), np.array([0, 1]), 4, freq=f, sigma=sigma, seg_width=w)
    analytic = sigma * w ** 2 * (2.0 * math.pi * f) ** 2 * Bm ** 2 / 24.0
    assert abs(q[0] - analytic) / analytic < 0.015
    assert q[0] == q[1]
    # f=0 ⇒ нет потерь
    q0 = magnet_eddy_loss_density_from_waveform(
        _sinusoid(N, Bm, n_probes=2), np.array([0, 1]), 4, freq=0.0, sigma=sigma, seg_width=w)
    assert q0.max() == 0.0


def test_eddy_frequency_squared_scaling():
    # гармоника k=2 = поле удвоенной частоты ⇒ потери ×4 (∝ f²)
    kw = dict(freq=500.0, sigma=1.0e6, seg_width=2.0e-3)
    q1 = magnet_eddy_loss_density_from_waveform(_sinusoid(128, 0.04, k=1), np.array([0]), 1, **kw)
    q2 = magnet_eddy_loss_density_from_waveform(_sinusoid(128, 0.04, k=2), np.array([0]), 1, **kw)
    assert 3.85 < q2[0] / q1[0] < 4.15


def test_segmentation_reduces_as_inverse_square():
    B = _sinusoid(64, 0.05)
    kw = dict(freq=800.0, sigma=1.0e6)
    q_full = magnet_eddy_loss_density_from_waveform(B, np.array([0]), 1, seg_width=4.0e-3, **kw)
    q_half = magnet_eddy_loss_density_from_waveform(B, np.array([0]), 1, seg_width=2.0e-3, **kw)
    assert abs(q_full[0] / q_half[0] - 4.0) < 1e-9      # w→w/2 ⇒ ×1/4


def test_conductivity_ratio_smco_vs_ndfeb():
    # та же волна, разная σ ⇒ потери ∝ σ. SmCo проводнее NdFeB (~1.6×) ⇒ больше вихревых.
    B = _sinusoid(64, 0.05)
    kw = dict(freq=800.0, seg_width=3.0e-3)
    q_nd = magnet_eddy_loss_density_from_waveform(B, np.array([0]), 1, sigma=0.79e6, **kw)
    q_sm = magnet_eddy_loss_density_from_waveform(B, np.array([0]), 1, sigma=1.18e6, **kw)
    assert abs(q_sm[0] / q_nd[0] - 1.18 / 0.79) < 1e-6


def test_segment_width_helper():
    p = OutrunnerPMSMParams(n_slots=12, n_poles=14)
    w1 = magnet_segment_width(p, 1)
    w2 = magnet_segment_width(p, 2)
    assert w1 > 0.0
    assert abs(w1 / w2 - 2.0) < 1e-9


def test_rotor_side_loss_density_real_sweep():
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.006)
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    steel = m270_35a_bh_curve()
    q, f_ref = magnet_rotor_loss_density(
        params, magnet, steel, speed_rpm=7000.0, sigma_pm=0.79e6,
        magnet_seg_width=magnet_segment_width(params, 1),
        mech_span=2.0 * math.pi / params.n_slots,   # зубцовый пролёт (дёшево, доминир. пульсация)
        n_positions=8, relaxation=0.1, max_iter=200,
    )
    geo = build_outrunner_spm_pmsm(params)
    rotor_side = np.isin(geo.region, (int(Region.MAGNET), int(Region.ROTOR_YOKE)))

    assert q.shape == (geo.mesh.n_cells,)
    assert f_ref > 0.0
    assert np.all(np.isfinite(q))
    assert np.any(q[geo.region == int(Region.MAGNET)] > 0.0)     # вихревые магнита есть
    assert np.allclose(q[~rotor_side], 0.0)                       # ноль вне ротор-стороны
