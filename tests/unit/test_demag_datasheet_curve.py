from __future__ import annotations

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_curves import (
    DemagnetizationCurveBH,
    demag_curve_from_datasheet,
    demag_curve_from_datasheet_cgs,
    _KOE_TO_A_PER_M,
)
from magcore.domain.magnet_model import n42sh_magnet


def _user_curve():
    # Excel-кейс пользователя: Br=11114 Гс, HcB=10.4, Hk=15.6, HcJ=28.4 кЭ.
    return demag_curve_from_datasheet_cgs("u", "user N42-like", 11114.0, 10.4, 15.6, 28.4)


def test_remanence_and_normal_coercivity() -> None:
    c = _user_curve()
    Br = 11114.0 * 1e-4
    Hcb = 10.4 * _KOE_TO_A_PER_M
    assert c.B_of_H(0.0) == pytest.approx(Br, rel=1e-6)
    # нормальная коэрцитивность: B=0 на recoil-линии при H = -HcB
    assert c.B_of_H(-Hcb) == pytest.approx(0.0, abs=2e-3)


def test_recoil_slope_is_mu0_mu_rec() -> None:
    c = _user_curve()
    Br = 11114.0 * 1e-4
    Hcb = 10.4 * _KOE_TO_A_PER_M
    mu_rec = Br / (MU0 * Hcb)
    # наклон recoil-прямой выше колена = mu0*mu_rec
    H1, H2 = -1.0e5, -2.0e5  # оба выше колена (-1.24e6)
    slope = (c.B_of_H(H2) - c.B_of_H(H1)) / (H2 - H1)
    assert slope == pytest.approx(MU0 * mu_rec, rel=1e-3)


def test_intrinsic_at_knee_matches_spreadsheet() -> None:
    c = _user_curve()
    Hk = 15.6 * _KOE_TO_A_PER_M
    # J(колено) = B(-Hk) - mu0*(-Hk); таблица даёт ~1.0043 Тл (=10043 Гс из шаблона)
    J_knee = c.B_of_H(-Hk) - MU0 * (-Hk)
    assert J_knee == pytest.approx(1.0043, abs=3e-3)


def test_C1_continuity_of_slope_at_knee() -> None:
    c = _user_curve()
    Hk = 15.6 * _KOE_TO_A_PER_M
    s_above = c.slope_dBdH(-Hk + 5.0e4)
    s_below = c.slope_dBdH(-Hk - 5.0e4)
    # касательная парабола => наклоны по обе стороны колена близки (C^1)
    assert s_below == pytest.approx(s_above, rel=0.08)


def test_curve_monotone_increasing_in_H() -> None:
    c = _user_curve()
    assert np.all(np.diff(c.B_values) >= 0.0)
    assert np.all(np.diff(c.H_values) > 0.0)


def test_si_factory_basic() -> None:
    c = demag_curve_from_datasheet("x", "x", Br=1.2, Hcb=9.0e5, Hk=1.0e6, Hcj=1.8e6)
    assert c.B_of_H(0.0) == pytest.approx(1.2, rel=1e-6)
    mu_rec = 1.2 / (MU0 * 9.0e5)
    assert c.B_of_H(-9.0e5) == pytest.approx(0.0, abs=2e-3)
    # ниже колена нормальная кривая уходит в минус
    assert c.B_of_H(-1.8e6) < 0.0


def test_validation_rejects_bad_ordering() -> None:
    with pytest.raises(ValueError):  # Hk >= Hcj
        demag_curve_from_datasheet("x", "x", Br=1.2, Hcb=9.0e5, Hk=1.8e6, Hcj=1.0e6)
    with pytest.raises(ValueError):  # Br <= 0
        demag_curve_from_datasheet("x", "x", Br=-1.0, Hcb=9.0e5, Hk=1.0e6, Hcj=1.8e6)


def test_cgs_conversion_consistent_with_si() -> None:
    c_cgs = demag_curve_from_datasheet_cgs("u", "u", 11114.0, 10.4, 15.6, 28.4)
    c_si = demag_curve_from_datasheet(
        "u", "u",
        Br=11114.0 * 1e-4,
        Hcb=10.4 * _KOE_TO_A_PER_M,
        Hk=15.6 * _KOE_TO_A_PER_M,
        Hcj=28.4 * _KOE_TO_A_PER_M,
    )
    assert np.allclose(c_cgs.B_values, c_si.B_values)
    assert np.allclose(c_cgs.H_values, c_si.H_values)


# ---------------------------------------------------------------------------
# ФАКТИЧЕСКИЙ H_cB (нуль нормальной кривой) vs даташит-ПАРАМЕТР Hcb.
# Параметр задаёт лишь наклон mu_rec = Br/(mu0*Hcb) — «где был бы ноль, если бы
# прямой участок шёл НЕ ЛОМАЯСЬ». Пока колено ЗА H_cB, оба числа совпадают; как
# только колено заходит ПЕРЕД H_cB (нагрев), кривая ломается раньше.
# ---------------------------------------------------------------------------


def test_hcb_actual_equals_parameter_when_knee_is_beyond() -> None:
    """Колено (15.6 кЭ) ЗА H_cB (10.4 кЭ) ⇒ прямая доходит до нуля неломаясь."""
    c = _user_curve()
    Hcb = 10.4 * _KOE_TO_A_PER_M
    assert c.Hcb_actual() == pytest.approx(Hcb, rel=1e-4)
    assert c.B_of_H(-c.Hcb_actual()) == pytest.approx(0.0, abs=1e-6)


def test_hcb_actual_is_smaller_when_knee_comes_first() -> None:
    """Колено ПЕРЕД H_cB ⇒ кривая ломается вниз и ноль наступает раньше параметра."""
    Br, Hcb_par, Hk, Hcj = 1.1, 7.9e5, 3.9e5, 4.6e5      # Hk < Hcb_par (горячий магнит)
    c = demag_curve_from_datasheet("hot", "hot", Br=Br, Hcb=Hcb_par, Hk=Hk, Hcj=Hcj)
    assert c.Hcb_actual() < Hcb_par                        # параметр завышает
    assert Hk < c.Hcb_actual() < Hcj                       # ноль между коленом и H_cJ
    assert c.B_of_H(-c.Hcb_actual()) == pytest.approx(0.0, abs=1e-6)


def test_hcb_actual_collapses_faster_than_Br_on_heating() -> None:
    """Физический оракул: у NdFeB H_cB с нагревом падает кратно быстрее, чем B_r."""
    mag = n42sh_magnet((1.0, 0.0, 0.0))
    c20, c150 = mag.curve_at(20.0), mag.curve_at(150.0)
    # при 20 °C колено (17 кЭ) ЗА H_cB (11.6 кЭ) ⇒ параметр и кривая совпадают
    assert c20.Hcb_actual() == pytest.approx(mag.Hcb(20.0), rel=1e-4)
    # при 150 °C колено заходит вперёд ⇒ параметр уже НЕ равен нулю кривой
    assert c150.Hcb_actual() < 0.75 * mag.Hcb(150.0)
    drop_Hcb = 1.0 - c150.Hcb_actual() / c20.Hcb_actual()
    drop_Br = 1.0 - mag.Br(150.0) / mag.Br(20.0)
    assert drop_Hcb > 2.0 * drop_Br


def test_hcb_actual_rejects_curve_without_zero_crossing() -> None:
    with pytest.raises(ValueError):        # вся кривая выше нуля
        DemagnetizationCurveBH("p", "p", np.array([-1.0e5, 0.0]), np.array([0.5, 1.0])).Hcb_actual()
    with pytest.raises(ValueError):        # вся кривая ниже нуля
        DemagnetizationCurveBH("n", "n", np.array([-1.0e5, 0.0]), np.array([-2.0, -1.0])).Hcb_actual()
