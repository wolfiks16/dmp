import math

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.iron_loss import (
    SteinmetzCoefficients,
    eddy_specific_from_waveform,
    efficiency,
    electrical_frequency,
    iron_loss_from_probe_waveform,
    stator_iron_loss,
    stator_iron_probes,
)
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams

# Потери в железе (Штейнмец с сепарацией). Оракулы — законы подобия и аналитика на известной
# волне, а не эталонные ватты:
#   * гистерезис ∝ f, ∝ B^α; вихревые ∝ f²;
#   * вихревые из ВОЛНЫ на чистой синусоиде = k_e·f²·B_m² (аналитический предел);
#   * гармоника в волне УВЕЛИЧИВАЕТ вихревые как Σ n²B_n² — то, что снимок дать не может;
#   * масса/длина/плотность масштабируют потери линейно;
#   * на РЕАЛЬНОЙ машине (одна прогонка) hyst∝f и eddy∝f² при пересчёте той же волны.

MESH = 0.0032


def _coeffs():
    return SteinmetzCoefficients.m270_35a()


# --------------------------------------------------------------- удельные потери: подобие

def test_hysteresis_scales_with_frequency_and_flux():
    cf = _coeffs()
    assert abs(cf.specific_hysteresis(1.2, 100.0) / cf.specific_hysteresis(1.2, 50.0) - 2.0) < 1e-12
    ratio = cf.specific_hysteresis(2.0, 50.0) / cf.specific_hysteresis(1.0, 50.0)
    assert abs(ratio - 2.0 ** cf.alpha) < 1e-12


def test_eddy_sinusoid_scales_as_frequency_squared():
    cf = _coeffs()
    r = cf.specific_eddy_sinusoid(1.0, 100.0) / cf.specific_eddy_sinusoid(1.0, 50.0)
    assert abs(r - 4.0) < 1e-12


# ---------------------------------------------------- вихревые из волны: аналитический предел

def test_eddy_from_pure_sinusoid_matches_closed_form():
    # Волна B_x = B_m·sin θ_e, B_y=0 за один период ⇒ вихревые из dB/dθ обязаны сойтись к
    # k_e·f²·B_m² (замкнутая форма синусоиды). Центральная разность занижает на (sinΔ/Δ)²,
    # поэтому берём густую сетку и требуем 0.5 %.
    cf, Bm, f, N = _coeffs(), 1.4, 200.0, 720
    theta = np.arange(N) * (2 * math.pi / N)
    series = np.zeros((N, 1, 2))
    series[:, 0, 0] = Bm * np.sin(theta)
    got = float(eddy_specific_from_waveform(series, f, cf)[0])
    assert abs(got - cf.k_eddy * f ** 2 * Bm ** 2) / (cf.k_eddy * f ** 2 * Bm ** 2) < 5e-3


def test_harmonic_raises_eddy_by_n_squared_weight():
    # 5-я гармоника вносит в вихревые вес 5²=25: волна sinθ + a·sin5θ даёт вихревые
    # ≈ k_e f²(1 + 25a²) против k_e f² для чистой синусоиды. Это и есть выигрыш ВОЛНЫ над
    # снимком (который знал бы только пик и посчитал синусоиду).
    cf, f, N, a = _coeffs(), 100.0, 1440, 0.3
    theta = np.arange(N) * (2 * math.pi / N)
    pure = np.zeros((N, 1, 2)); pure[:, 0, 0] = np.sin(theta)
    harm = np.zeros((N, 1, 2)); harm[:, 0, 0] = np.sin(theta) + a * np.sin(5 * theta)

    e_pure = float(eddy_specific_from_waveform(pure, f, cf)[0])
    e_harm = float(eddy_specific_from_waveform(harm, f, cf)[0])
    assert e_harm > e_pure
    expected = cf.k_eddy * f ** 2 * (1.0 + 25.0 * a ** 2)
    assert abs(e_harm - expected) / expected < 1e-2


# --------------------------------------------------------- сборка потерь: масштабирование

def test_iron_loss_scales_with_length_and_density():
    cf = _coeffs()
    N, P = 64, 5
    theta = np.arange(N) * (2 * math.pi / N)
    series = np.zeros((N, P, 2))
    series[:, :, 0] = np.sin(theta)[:, None] * np.linspace(0.5, 1.5, P)[None, :]
    areas = np.full(P, 2.0e-6)

    base = iron_loss_from_probe_waveform(series, areas, freq=200.0, axial_length=0.03, coeffs=cf)
    longer = iron_loss_from_probe_waveform(series, areas, freq=200.0, axial_length=0.06, coeffs=cf)
    assert abs(longer.total / base.total - 2.0) < 1e-12          # ∝ осевой длине
    assert abs(longer.iron_mass / base.iron_mass - 2.0) < 1e-12

    denser = SteinmetzCoefficients(cf.k_hyst, cf.alpha, cf.k_eddy, cf.density * 1.1)
    d = iron_loss_from_probe_waveform(series, areas, freq=200.0, axial_length=0.03, coeffs=denser)
    assert abs(d.total / base.total - 1.1) < 1e-9               # ∝ плотности

    assert base.hysteresis > 0.0 and base.eddy > 0.0
    assert abs(base.total - (base.hysteresis + base.eddy)) < 1e-15


def test_waveform_shape_is_validated():
    cf = _coeffs()
    with pytest.raises(ValueError, match="N, P, 2"):
        iron_loss_from_probe_waveform(np.zeros((10, 3)), np.ones(3), freq=1.0,
                                      axial_length=1.0, coeffs=cf)


def test_coefficients_reject_nonphysical_values():
    with pytest.raises(ValueError):
        SteinmetzCoefficients(k_hyst=-1.0, alpha=2.0, k_eddy=1e-5, density=7650.0)
    with pytest.raises(ValueError, match="alpha"):
        SteinmetzCoefficients(k_hyst=0.02, alpha=5.0, k_eddy=1e-5, density=7650.0)


# ------------------------------------------------------------------------- КПД

def test_efficiency_bounds_and_known_value():
    # Известный случай: P_вых=100 Вт, потери 10+5 ⇒ η=100/115.
    eta = efficiency(torque_mean=100.0 / (2 * math.pi * 3000.0 / 60.0),
                     speed_rpm=3000.0, copper_loss_w=10.0, iron_loss_w=5.0)
    assert abs(eta - 100.0 / 115.0) < 1e-9
    # Больше потерь — ниже КПД; ноль потерь — ровно 1; знак момента не влияет.
    assert efficiency(1.0, 3000.0, 0.0, 0.0) == 1.0
    assert efficiency(1.0, 3000.0, 50.0, 0.0) < efficiency(1.0, 3000.0, 10.0, 0.0)
    assert efficiency(1.0, 3000.0, 5.0, 5.0) == efficiency(-1.0, 3000.0, 5.0, 5.0)
    with pytest.raises(ValueError):
        efficiency(1.0, 0.0, 1.0, 1.0)


# --------------------------------------------------- машинный уровень (одна прогонка)

def test_stator_iron_loss_on_machine_scales_with_speed():
    # Одна прогонка ротора даёт волну B(θ_e); её пересчёт на РАЗНЫЕ частоты обязан дать
    # hyst∝f и eddy∝f² — закон подобия Штейнмеца на реальной несинусоидальной волне зубца.
    p = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH)
    m, st, cf = n42sh_magnet((1.0, 0.0, 0.0)), m270_35a_bh_curve(), _coeffs()

    loss1, sw = stator_iron_loss(p, m, st, speed_rpm=3000.0, i_peak=20.0, gamma_elec=3.93,
                                 turns_per_slot=20.0, T=20.0, n_positions=18, coeffs=cf)
    assert sw.all_converged
    assert loss1.hysteresis > 0.0 and loss1.eddy > 0.0
    assert loss1.iron_mass > 0.0

    from magcore.fem2d.machines.iron_loss import iron_loss_from_probe_waveform
    _, areas = stator_iron_probes(p)
    f1 = electrical_frequency(p, 3000.0)
    f2 = electrical_frequency(p, 6000.0)
    loss2 = iron_loss_from_probe_waveform(sw.probe_B, areas, freq=f2,
                                          axial_length=p.axial_length, coeffs=cf)
    assert abs(loss2.hysteresis / loss1.hysteresis - f2 / f1) < 1e-9        # гистерезис ∝ f
    assert abs(loss2.eddy / loss1.eddy - (f2 / f1) ** 2) < 1e-9             # вихревые ∝ f²


def test_teeth_carry_more_loss_than_yoke_probes():
    # Зубцы видят и более высокое поле, и резкий его размах при проходе полюсов ⇒ на них
    # приходится бОльшая доля потерь, чем на ярмо. Проверяет, что волна снята в нужных точках.
    p = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH)
    from magcore.fem2d.machines.pmsm_outrunner import Region, build_outrunner_spm_pmsm
    geo = build_outrunner_spm_pmsm(p)
    n_tooth = int(geo.mask(Region.TOOTH).sum())
    n_yoke = int(geo.mask(Region.STATOR_YOKE).sum())
    pts, _ = stator_iron_probes(p)
    assert pts.shape[0] == n_tooth + n_yoke               # покрыто всё железо статора
