import numpy as np

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.iron_loss import (
    SteinmetzCoefficients,
    electrical_frequency,
    iron_loss_density_from_waveform,
    stator_iron_loss_density,
)
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)

# P-B3: поэлементная плотность потерь в стали статора q_Fe [Вт/м³] (источник тепла).
# Оракулы: сверка Cogent-коэффициентов с datasheet; масштаб f/f²; per-cell из синт. волны;
# реальный sweep — q>0 только в железе статора, растёт со скоростью.


def test_cogent_coefficients_match_datasheet():
    cf = SteinmetzCoefficients.m270_35a_cogent()
    # Cogent: 1.0 Тл/50 Гц ≈ 1.01 Вт/кг; 1.5 Тл/50 Гц ≈ 2.47 Вт/кг (фит ±~20 % на точке).
    p10 = float(cf.specific_hysteresis(1.0, 50.0) + cf.specific_eddy_sinusoid(1.0, 50.0))
    p15 = float(cf.specific_hysteresis(1.5, 50.0) + cf.specific_eddy_sinusoid(1.5, 50.0))
    assert 0.8 < p10 < 1.3
    assert 2.0 < p15 < 3.0
    # f=0 ⇒ нет потерь
    assert float(cf.specific_hysteresis(1.5, 0.0) + cf.specific_eddy_sinusoid(1.5, 0.0)) == 0.0


def test_loss_frequency_scaling():
    cf = SteinmetzCoefficients.m270_35a_cogent()
    # гистерезис ∝ f; вихревые ∝ f²
    assert abs(cf.specific_hysteresis(1.0, 100.0) / cf.specific_hysteresis(1.0, 50.0) - 2.0) < 1e-9
    assert abs(cf.specific_eddy_sinusoid(1.0, 100.0) / cf.specific_eddy_sinusoid(1.0, 50.0) - 4.0) < 1e-9


def test_density_from_synthetic_sinusoid():
    cf = SteinmetzCoefficients.m270_35a_cogent()
    N, freq, Bm = 48, 400.0, 1.5
    theta = 2.0 * np.pi * np.arange(N) / N
    probe_idx = np.array([2, 6])
    B = np.zeros((N, probe_idx.size, 2))
    B[:, :, 0] = (Bm * np.sin(theta))[:, None]              # чистая синусоида |B|=1.5
    q = iron_loss_density_from_waveform(B, probe_idx, n_cells=10, freq=freq, coeffs=cf)

    expected = float(cf.specific_hysteresis(Bm, freq) + cf.specific_eddy_sinusoid(Bm, freq)) * cf.density
    assert abs(q[2] - expected) / expected < 0.03       # волновые вихревые ≈ закрытая форма
    assert q[2] == q[6]
    assert np.count_nonzero(q) == 2                      # ноль вне пробных ячеек


def test_stator_iron_density_real_sweep():
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.005)
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    steel = m270_35a_bh_curve()

    q7000, sweep = stator_iron_loss_density(
        params, magnet, steel, speed_rpm=7000.0, n_positions=6, relaxation=0.1, max_iter=200,
    )
    geo = build_outrunner_spm_pmsm(params)
    stator = np.isin(geo.region, (int(Region.STATOR_YOKE), int(Region.TOOTH)))

    assert q7000.shape == (geo.mesh.n_cells,)
    assert np.any(q7000[stator] > 0.0)                   # потери в железе статора есть
    assert np.allclose(q7000[~stator], 0.0)              # ноль вне железа статора

    # та же волна при меньшей скорости ⇒ меньше потерь (p_h∝f, p_e∝f²)
    idx = np.where(stator)[0]
    q3500 = iron_loss_density_from_waveform(
        sweep.probe_B, idx, geo.mesh.n_cells,
        freq=electrical_frequency(params, 3500.0), coeffs=SteinmetzCoefficients.m270_35a_cogent(),
    )
    assert q7000.sum() > q3500.sum()
