import math

import numpy as np

from magcore.constants import MU0
from magcore.fem2d.machines.iron_loss import (
    STEEL10_DENSITY,
    STEEL10_RESISTIVITY,
    SteinmetzCoefficients,
)
from magcore.fem2d.machines.magnet_loss import (
    effective_eddy_thickness,
    magnet_eddy_loss_density_from_waveform,
    skin_depth,
    solid_steel_loss_density_from_waveform,
)

# Сталь 10: шихтованный статор (Штейнмец, k_c ВЫВЕДЕН аналитически) + массивный ротор
# (модель сплошного тела со СКИН-ПРЕДЕЛОМ). Оракулы физики, не подгонки.


def _sinusoid(N, Bm, n_probes=1):
    theta = 2.0 * math.pi * np.arange(N) / N
    B = np.zeros((N, n_probes, 2))
    B[:, :, 0] = (Bm * np.sin(theta))[:, None]
    return B


def test_analytic_eddy_coefficient_reproduces_m270_fit():
    # k_c = π²d²/(6ρρ_m) — ФИЗИКА. Сверка: для M270 (0.35мм, 0.52µΩм, 7690) должно совпасть
    # с коэффициентом из фита к таблице Cogent (5.04e-5) — независимое подтверждение обоих.
    cf = SteinmetzCoefficients.from_lamination(
        k_hyst=1.8e-2, alpha=1.67, thickness=0.35e-3, resistivity=0.52e-6, density=7690.0)
    assert abs(cf.k_eddy - 5.04e-5) / 5.04e-5 < 0.02


def test_steel10_laminated_losses_exceed_m270():
    s10 = SteinmetzCoefficients.steel10_laminated(0.5e-3)
    m270 = SteinmetzCoefficients.m270_35a_cogent()
    # Вихревые ∝ d²/ρ. Сравнивать надо АНАЛИТИКУ с АНАЛИТИКОЙ: у `m270_35a_cogent` k_eddy —
    # из 2-членного ФИТА (6.9e-5, вобрал избыточные потери), а не физический 5.04e-5.
    m270_analytic = SteinmetzCoefficients.from_lamination(
        k_hyst=1.83e-2, alpha=1.67, thickness=0.35e-3, resistivity=0.52e-6, density=7690.0)
    assert abs(s10.k_eddy / m270_analytic.k_eddy - (0.5 / 0.35) ** 2 * (0.52 / 0.14)) < 0.2
    assert s10.k_eddy > 4.0 * m270.k_eddy          # и против фитованного — кратно выше
    # суммарные удельные потери при 1.5 Тл/50 Гц в литературном коридоре 10–15 Вт/кг
    p = float(s10.specific_hysteresis(1.5, 50.0) + s10.specific_eddy_sinusoid(1.5, 50.0))
    assert 9.0 < p < 15.0
    # и заметно выше электротехнической стали
    p_m = float(m270.specific_hysteresis(1.5, 50.0) + m270.specific_eddy_sinusoid(1.5, 50.0))
    assert p > 3.0 * p_m
    assert s10.density == STEEL10_DENSITY


def test_skin_depth_and_effective_thickness():
    sigma = 1.0 / STEEL10_RESISTIVITY
    d = skin_depth(1400.0, sigma, 1000.0)
    assert 0.1e-3 < d < 0.3e-3                     # ≈0.16 мм для Ст10 при 1400 Гц
    # массивное ярмо 3 мм: скин ОГРАНИЧИВАЕТ (w_eff = 2δ ≪ w)
    w_eff = effective_eddy_thickness(3.0e-3, 1400.0, sigma, 1000.0)
    assert abs(w_eff - 2.0 * d) < 1e-12
    assert w_eff < 3.0e-3 / 4.0
    # магнит (μ_r≈1.05, σ≈0.79e6): δ ≫ ширины полюса ⇒ предел НЕ срабатывает
    w_mag = effective_eddy_thickness(9.0e-3, 1400.0, 0.79e6, 1.05)
    assert w_mag == 9.0e-3
    # статика: δ→∞ ⇒ предела нет
    assert effective_eddy_thickness(3.0e-3, 0.0, sigma, 1000.0) == 3.0e-3


def test_solid_rotor_loss_is_bounded_by_skin_effect():
    # Наивная формула (w=3мм, без скин-предела) ЗАВЫШАЕТ на ~2 порядка — тест фиксирует,
    # что модель этого не делает (иначе нагрев ротора был бы фантастическим).
    sigma = 1.0 / STEEL10_RESISTIVITY
    cf = SteinmetzCoefficients.steel10_laminated(0.5e-3)
    B = _sinusoid(64, 0.05)
    q_model = solid_steel_loss_density_from_waveform(
        B, np.array([0]), 1, freq=1400.0, sigma=sigma, thickness=3.0e-3, mu_r=1000.0, coeffs=cf)
    q_naive = magnet_eddy_loss_density_from_waveform(
        B, np.array([0]), 1, freq=1400.0, sigma=sigma, seg_width=3.0e-3)
    assert q_model[0] < q_naive[0] / 20.0
    assert q_model[0] > 0.0
    # гистерезис включён: без вихревых (σ→0) остаётся положительный вклад
    q_h = solid_steel_loss_density_from_waveform(
        B, np.array([0]), 1, freq=1400.0, sigma=1e-9, thickness=3.0e-3, mu_r=1000.0, coeffs=cf)
    assert q_h[0] > 0.0


def test_solid_rotor_worse_than_laminated_but_not_absurd():
    # Инженерный итог: массивная Ст10 хуже шихтованной M270 в единицы раз (не в сотни).
    sigma10 = 1.0 / STEEL10_RESISTIVITY
    w_eff = effective_eddy_thickness(3.0e-3, 1400.0, sigma10, 1000.0)
    solid = sigma10 * w_eff ** 2
    lam_m270 = (1.0 / 0.52e-6) * (0.35e-3) ** 2
    assert 1.5 < solid / lam_m270 < 8.0
