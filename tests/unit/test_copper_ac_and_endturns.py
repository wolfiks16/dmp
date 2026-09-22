# -*- coding: utf-8 -*-
"""
ПОТЕРИ В МЕДИ: лобовые части и переменная составляющая (Дауэлл) + кремнистая сталь.

Добавлено по итогам сверки с Scorpion IM-8008 (2026-09-10): модель потерь опровергнута
в двух местах — гистерезис статора завышен (считался из Стали 10), а на высоких частотах
недоставало потерь (медь без лобовых частей и без переменной составляющей). Каждая новая
формула проверяется независимо — предельными случаями и выводом «вручную», а не сверкой
с ней же самой.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.fem2d.losses import (
    CU_RHO0,
    copper_resistivity,
    copper_skin_depth,
    dowell_ac_factor,
    round_wire_delta,
)
from magcore.fem2d.machines.characteristics import phase_resistance, tooth_coil_end_length
from magcore.fem2d.machines.excitation import slot_areas
from magcore.fem2d.machines.iron_loss import (
    M270_K_HYST,
    STEEL10_K_HYST_LIT,
    SteinmetzCoefficients,
)
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, build_outrunner_spm_pmsm


# ======================================================================== глубина проникновения

def test_skin_depth_matches_hand_value() -> None:
    """δ(1 кГц, 20 °C) = √(ρ/(π·f·μ₀)) ≈ 2,09 мм — считаем независимо."""
    hand = math.sqrt(CU_RHO0 / (math.pi * 1000.0 * MU0))
    assert copper_skin_depth(1000.0, 20.0) == pytest.approx(hand, rel=1e-12)
    assert hand == pytest.approx(2.09e-3, rel=5e-3)


def test_skin_depth_scales_as_inverse_root_frequency() -> None:
    assert copper_skin_depth(4000.0) == pytest.approx(0.5 * copper_skin_depth(1000.0), rel=1e-12)


def test_skin_depth_grows_with_temperature() -> None:
    """Горячая медь хуже проводит ⇒ поле проникает глубже."""
    ratio = copper_skin_depth(1000.0, 120.0) / copper_skin_depth(1000.0, 20.0)
    assert ratio == pytest.approx(math.sqrt(float(copper_resistivity(120.0)) / CU_RHO0), rel=1e-12)


# ======================================================================== коэффициент Дауэлла

@pytest.mark.parametrize("m", [1, 3, 9, 30])
def test_dowell_is_unity_at_dc(m: int) -> None:
    assert dowell_ac_factor(0.0, m) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("m", [1, 4, 9, 27])
@pytest.mark.parametrize("delta", [0.02, 0.05, 0.1])
def test_dowell_matches_low_frequency_expansion(m: int, delta: float) -> None:
    """Точная формула обязана переходить в 1 + (5m²−1)/45·Δ⁴ — независимый вывод."""
    series = 1.0 + (5.0 * m * m - 1.0) / 45.0 * delta ** 4
    assert dowell_ac_factor(delta, m) == pytest.approx(series, rel=2e-6)


@pytest.mark.parametrize("edge", [1.0e-2, 30.0])
def test_dowell_is_continuous_across_branch_switches(edge: float) -> None:
    """Ветви «разложение / точная / асимптотика» сшиты без скачка."""
    for m in (1, 9):
        lo = dowell_ac_factor(edge * (1 - 1e-9), m)
        hi = dowell_ac_factor(edge * (1 + 1e-9), m)
        assert lo == pytest.approx(hi, rel=1e-6)


def test_dowell_single_layer_tends_to_pure_skin_effect() -> None:
    """Один слой (m = 1) при толстом проводнике: F_R → Δ (ток вытеснен в слой δ)."""
    assert dowell_ac_factor(12.0, 1) == pytest.approx(12.0, rel=1e-6)


def test_dowell_is_monotone_in_delta_and_in_layers() -> None:
    d = np.linspace(0.0, 3.0, 61)
    for m in (1, 5, 20):
        f = dowell_ac_factor(d, m)
        assert np.all(np.diff(f) >= -1e-12)
    for delta in (0.1, 0.5, 1.5):
        assert dowell_ac_factor(delta, 2) < dowell_ac_factor(delta, 5) < dowell_ac_factor(delta, 20)


def test_dowell_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        dowell_ac_factor(0.1, 0)
    with pytest.raises(ValueError):
        dowell_ac_factor(-0.1, 3)


def test_round_wire_delta_uses_equivalent_foil() -> None:
    """Круглый провод d → фольга h = (√π/2)·d; при d = δ и η = 1 получаем Δ = √π/2."""
    f = 1500.0
    d = copper_skin_depth(f)
    assert round_wire_delta(strand_diameter=d, freq=f, porosity=1.0) == pytest.approx(
        0.5 * math.sqrt(math.pi), rel=1e-12)


# ======================================================================== лобовые части

@pytest.fixture(scope="module")
def geo():
    return build_outrunner_spm_pmsm(OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.004))


def test_end_length_matches_hand_formula(geo) -> None:
    """π·c, c = R_ср·(θ_зуба + θ_паза/2) — сторона катушки занимает полпаза у своего зуба."""
    p = geo.params
    pitch = 2.0 * math.pi / p.n_slots
    t, s = p.tooth_width_frac * pitch, (1.0 - p.tooth_width_frac) * pitch
    c = 0.5 * (p.R_sy + p.R_s_out) * (t + 0.5 * s)
    assert tooth_coil_end_length(geo) == pytest.approx(math.pi * c, rel=1e-12)


def test_phase_resistance_default_is_unchanged(geo) -> None:
    """СТРАЖ ОБРАТНОЙ СОВМЕСТИМОСТИ: без новых аргументов — ровно прежняя формула."""
    N, k, T = 20.0, 0.45, 60.0
    L = geo.params.axial_length
    old = float((float(copper_resistivity(T)) * L * N ** 2
                 / (k * np.asarray(slot_areas(geo)))).sum() / 3.0)
    assert phase_resistance(geo, turns_per_slot=N, slot_fill=k, T=T) == pytest.approx(old, rel=1e-14)


def test_end_turns_scale_resistance_exactly(geo) -> None:
    """Проводник паза несёт половину лобовой части витка ⇒ R ∝ (L + l_лоб/2)."""
    L = geo.params.axial_length
    l_end = tooth_coil_end_length(geo)
    r0 = phase_resistance(geo, turns_per_slot=10.0, slot_fill=0.45, T=20.0)
    r1 = phase_resistance(geo, turns_per_slot=10.0, slot_fill=0.45, T=20.0,
                          end_length_per_turn=l_end)
    assert r1 / r0 == pytest.approx((L + 0.5 * l_end) / L, rel=1e-12)


def test_ac_factor_applies_only_to_the_slot_part(geo) -> None:
    """Переменная составляющая — только в пазу: лобовые части в воздухе, к ним F_R не идёт."""
    L = geo.params.axial_length
    l_end, F = tooth_coil_end_length(geo), 1.37
    r0 = phase_resistance(geo, turns_per_slot=10.0, slot_fill=0.45, T=20.0)
    r = phase_resistance(geo, turns_per_slot=10.0, slot_fill=0.45, T=20.0,
                         end_length_per_turn=l_end, ac_factor=F)
    assert r / r0 == pytest.approx((F * L + 0.5 * l_end) / L, rel=1e-12)


def test_phase_resistance_rejects_nonphysical_inputs(geo) -> None:
    with pytest.raises(ValueError):
        phase_resistance(geo, turns_per_slot=10.0, slot_fill=0.45, T=20.0, ac_factor=0.9)
    with pytest.raises(ValueError):
        phase_resistance(geo, turns_per_slot=10.0, slot_fill=0.45, T=20.0, end_length_per_turn=-1e-3)


# ======================================================================== кремнистая сталь

def test_silicon_steel_eddy_term_is_derived_not_fitted() -> None:
    """Вихревой член — физика: при 0,35 мм обязан совпасть с фитом Cogent 5,04e-5."""
    assert SteinmetzCoefficients.silicon_steel_laminated(0.35e-3).k_eddy == pytest.approx(5.04e-5, rel=0.02)


def test_silicon_steel_eddy_scales_as_thickness_squared() -> None:
    a = SteinmetzCoefficients.silicon_steel_laminated(0.35e-3).k_eddy
    b = SteinmetzCoefficients.silicon_steel_laminated(0.20e-3).k_eddy
    assert b / a == pytest.approx((0.20 / 0.35) ** 2, rel=1e-12)


def test_silicon_steel_hysteresis_far_below_steel10() -> None:
    """Корень опровержения сверки: гистерезис кремнистой стали в ~5,5 раза ниже Стали 10."""
    si = SteinmetzCoefficients.silicon_steel_laminated()
    assert si.k_hyst == M270_K_HYST
    assert STEEL10_K_HYST_LIT / si.k_hyst > 5.0


def test_silicon_steel_specific_loss_near_datasheet() -> None:
    """1,0 Тл / 50 Гц на листе 0,35 мм — Cogent даёт 1,01 Вт/кг; ждём попадания в ±25 %."""
    cf = SteinmetzCoefficients.silicon_steel_laminated(0.35e-3)
    p = float(cf.specific_hysteresis(1.0, 50.0) + cf.specific_eddy_sinusoid(1.0, 50.0))
    assert 0.76 < p < 1.26
