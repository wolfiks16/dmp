from __future__ import annotations

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.steel_curves import SteelBHCurve, m270_35a_bh_curve


def test_m270_curve_constructs_and_validates() -> None:
    c = m270_35a_bh_curve()
    assert c.n_points >= 10
    assert c.B_max == pytest.approx(2.21)
    assert c.temperature_c is None


def test_nodes_round_trip_H_of_B() -> None:
    c = m270_35a_bh_curve()
    for h, b in zip(c.H_values, c.B_values):
        assert c.H_of_B(float(b)) == pytest.approx(float(h), abs=1e-6)


def test_chord_limit_at_zero_equals_initial_differential() -> None:
    c = m270_35a_bh_curve()
    assert c.nu_chord(0.0) == pytest.approx(c.nu_initial)
    assert c.nu_chord(1e-13) == pytest.approx(c.nu_initial)
    # первый сегмент через начало => хорда постоянна и равна nu_d(0) на (0, B_1]
    assert c.nu_chord(0.5 * float(c.B_values[1])) == pytest.approx(c.nu_initial, rel=1e-9)


def test_reluctivity_positive_and_bounded() -> None:
    c = m270_35a_bh_curve()
    nu_max = 1.0 / MU0
    for B in np.linspace(1e-6, 3.0, 200):
        nu = c.nu_chord(B)
        nud = c.nu_differential(B)
        assert nu > 0.0 and nud > 0.0
        assert nu <= nu_max * (1.0 + 1e-9)
        assert nud <= nu_max * (1.0 + 1e-9)


def test_saturation_tail_slope_is_mu0() -> None:
    c = m270_35a_bh_curve()
    # за B_max наклон H(B) равен 1/mu0 => nu_d = 1/mu0
    assert c.nu_differential(c.B_max + 1.0) == pytest.approx(1.0 / MU0)
    # H растёт как (B - B_max)/mu0
    dB = 0.5
    assert c.H_of_B(c.B_max + dB) == pytest.approx(c.H_max + dB / MU0, rel=1e-9)
    # хорда стремится к 1/mu0 при больших B
    assert c.nu_chord(50.0) == pytest.approx(1.0 / MU0, rel=0.1)


def test_H_of_B_monotone_increasing() -> None:
    c = m270_35a_bh_curve()
    Bs = np.linspace(0.0, 3.0, 300)
    Hs = np.array([c.H_of_B(B) for B in Bs])
    assert np.all(np.diff(Hs) >= 0.0)


def test_mu_r_shape_peak_and_saturation() -> None:
    c = m270_35a_bh_curve()
    # пик хордовой mu_r в разумном диапазоне для нонориент. стали
    mu_r = np.array([c.mu_r_chord(B) for B in np.linspace(0.05, 2.0, 100)])
    assert 2000.0 < mu_r.max() < 12000.0
    # дифф. проницаемость в глубоком насыщении -> mu0 (mu_r_diff -> 1)
    assert c.nu_differential(3.0) == pytest.approx(1.0 / MU0)
    # хордовая mu_r при больших B убывает к ~1 (медленно), но > 1
    assert 1.0 < c.mu_r_chord(3.0) < 5.0


def test_differential_at_node_uses_right_segment() -> None:
    c = m270_35a_bh_curve()
    # nu_d(0) — наклон первого сегмента
    assert c.nu_differential(0.0) == pytest.approx(float(c.H_values[1] / c.B_values[1]))


def test_chord_consistent_with_H_of_B() -> None:
    c = m270_35a_bh_curve()
    for B in [0.3, 0.8, 1.0, 1.5, 1.9, 2.1]:
        assert c.nu_chord(B) == pytest.approx(c.H_of_B(B) / B, rel=1e-12)


def test_validation_accepts_minimal_valid_curve() -> None:
    SteelBHCurve("ok", "ok", np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 2.0]))


def test_validation_rejects_non_origin_start() -> None:
    with pytest.raises(ValueError):
        SteelBHCurve("x", "x", np.array([10.0, 100.0]), np.array([0.1, 1.0]))


def test_validation_rejects_non_increasing_B() -> None:
    with pytest.raises(ValueError):
        SteelBHCurve("x", "x", np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 1.0]))


def test_validation_rejects_non_increasing_H() -> None:
    with pytest.raises(ValueError):
        SteelBHCurve("x", "x", np.array([0.0, 100.0, 100.0]), np.array([0.0, 1.0, 2.0]))


def test_validation_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        SteelBHCurve("x", "x", np.array([0.0]), np.array([0.0]))


def test_validation_rejects_superphysical_slope() -> None:
    # dH/dB = 2e6 > 1/mu0 (~7.96e5) => mu_diff < mu0, нефизично
    with pytest.raises(ValueError):
        SteelBHCurve("x", "x", np.array([0.0, 2.0e6]), np.array([0.0, 1.0]))
