from __future__ import annotations

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_curves import (
    demag_curve_from_datasheet,
    demag_curve_from_datasheet_cgs,
    _KOE_TO_A_PER_M,
)


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
