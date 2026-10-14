import math

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.excitation import phase_currents
from magcore.fem2d.machines.magnet_loss import (
    minimal_rotor_sweep,
    rotor_field_period,
    rotor_frame_B_series,
)
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, build_outrunner_spm_pmsm
from magcore.fem2d.machines.winding import star_of_slots_layout

# Л-34: пролёт свипа ротора. Волну в системе ротора дифференцируют периодической разностью, поэтому пролёт
# обязан быть кратен периоду поля в этой системе: на холостом ходу — зубцовому делению 2π/Z (ротор видит
# одинаковые зубцы), под нагрузкой — периоду картины токов пазов, увиденной с ротора. Прежние прогоны под
# нагрузкой брали зубцовое деление — стык давал ложную гармонику.
#
# ОРАКУЛ периода — картина токов пазов, увиденная с ротора, на ФАКТИЧЕСКИХ углах пазов из геометрии и на той
# же раскладке, что у свипа (`star_of_slots_layout`; ротор повёрнут на a, ток при γ + p·a — так считает
# `rotor_frame_B_series`). Без поля и без сетки: прямая проверка периодичности поля на перестраиваемой на
# каждом угле сетке тонет в шуме пересетки (Л-81).

COMBOS = [(12, 14), (36, 40), (12, 10), (9, 8), (18, 16), (24, 28)]


def _rotor_frame_currents(n_slots, n_poles, a, gamma=0.3):
    """Токи пазов по позиции паза, увиденной с ротора при угле a (позиция — номер зубцового деления)."""
    params = OutrunnerPMSMParams(n_slots=n_slots, n_poles=n_poles, mesh_size=0.008)
    geo = build_outrunner_spm_pmsm(params)
    sid = geo.slot_id
    cen = np.array([geo.mesh.cell_centroid(c) for c in range(geo.mesh.n_cells)])
    theta = np.array([math.atan2(*cen[sid == k].mean(axis=0)[::-1]) for k in range(n_slots)])
    layout = star_of_slots_layout(n_slots, n_poles)
    pitch = 2.0 * math.pi / n_slots
    current = layout.sign_of_slot * phase_currents(1.0, gamma + (n_poles // 2) * a)[layout.phase_of_slot]
    pos = np.mod(np.round((theta - a - theta[0]) / pitch).astype(int), n_slots)
    out = np.full(n_slots, np.nan)
    out[pos] = current
    assert not np.isnan(out).any()                      # каждая позиция занята ровно одним пазом
    return out


@pytest.mark.parametrize("n_slots,n_poles", COMBOS)
def test_loaded_period_is_where_the_rotor_frame_current_pattern_repeats(n_slots, n_poles):
    # Сдвиг ротора на m зубцовых делений с синхронным током: картина повторяется ровно при m, кратных
    # периоду, который считает `rotor_field_period`, — не реже и не чаще.
    pitch = 2.0 * math.pi / n_slots
    base = _rotor_frame_currents(n_slots, n_poles, 0.0)
    repeats = [m for m in range(1, n_slots + 1)
               if np.allclose(_rotor_frame_currents(n_slots, n_poles, m * pitch), base, rtol=0.0, atol=1e-12)]
    period = rotor_field_period(OutrunnerPMSMParams(n_slots=n_slots, n_poles=n_poles), loaded=True)
    step = int(round(period / pitch))
    assert abs(period - step * pitch) < 1e-12
    assert repeats == list(range(step, n_slots + 1, step))
    assert step > 1                                      # под нагрузкой зубцовое деление — НЕ период


@pytest.mark.parametrize("n_slots,n_poles", COMBOS[:-1])
def test_symmetric_winding_period_is_2pi_over_gcd_z_6p(n_slots, n_poles):
    # Теория для симметричной раскладки: токи паза — шесть осей через 60° (±A, ±B, ±C), фазор паза k + m
    # повёрнут на p·m·(2π/Z), поэтому картина повторяется, когда этот поворот кратен 60°: m·6p/Z — целое,
    # период 2π/НОД(Z, 6p). Прежнее правило Л-34 (2π/НОД(Z, 2p)) — кратное ему: верно, но не наименьшее.
    p = n_poles // 2
    period = rotor_field_period(OutrunnerPMSMParams(n_slots=n_slots, n_poles=n_poles), loaded=True)
    assert period == pytest.approx(2.0 * math.pi / math.gcd(n_slots, 6 * p), rel=1e-12)
    k = (2.0 * math.pi / math.gcd(n_slots, n_poles)) / period
    assert abs(k - round(k)) < 1e-9


def test_asymmetric_star_of_slots_layout_repeats_only_over_a_revolution():
    # Однослойная звезда пазов 24/28 разрешает «ничьи» неравномерно — картина токов повторяется только через
    # полный оборот, и формула 2π/НОД(Z, 2p) = 90° здесь дала бы НЕВЕРНЫЙ пролёт.
    p = OutrunnerPMSMParams(n_slots=24, n_poles=28)
    assert rotor_field_period(p, loaded=True) == pytest.approx(2.0 * math.pi)


def test_no_load_period_and_minimal_sweep():
    p = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.008)
    assert rotor_field_period(p, loaded=False) == pytest.approx(2.0 * math.pi / 12)
    # плотность выборки сохраняется: 12 положений на зубцовое деление ⇒ 24 на 60° под нагрузкой
    assert minimal_rotor_sweep(p, loaded=True, positions_per_tooth_pitch=12) == (pytest.approx(math.pi / 3), 24)
    assert minimal_rotor_sweep(p, loaded=False, positions_per_tooth_pitch=12) == (pytest.approx(math.pi / 6), 12)


def test_sweep_refuses_a_span_that_is_not_a_field_period():
    # Под нагрузкой зубцовый пролёт — отказ с причиной (проверка до решения поля); период, его кратные и на
    # холостом ходу зубцовое деление — допустимы (дальше обычный свип, здесь — из одного положения).
    p = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.008)
    kw = dict(speed_rpm=7000.0, cell_idx=np.array([0]), gamma_elec=0.0, n_positions=1)
    mag, steel = n42sh_magnet((1.0, 0.0, 0.0)), m270_35a_bh_curve()
    for bad in (2.0 * math.pi / 12, 0.7 * math.pi):
        with pytest.raises(ValueError, match="Л-34"):
            rotor_frame_B_series(p, mag, steel, i_peak=40.0, turns_per_slot=7.0, mech_span=bad, **kw)
    for span, ip in ((math.pi / 3, 40.0), (math.pi, 40.0), (2.0 * math.pi, 40.0), (2.0 * math.pi / 12, 0.0)):
        B, f = rotor_frame_B_series(p, mag, steel, i_peak=ip, turns_per_slot=7.0, mech_span=span, **kw)
        assert B.shape == (1, 1, 2) and f == pytest.approx(2.0 * math.pi * 7000.0 / 60.0 / span)
