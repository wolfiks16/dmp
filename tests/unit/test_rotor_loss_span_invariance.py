import math

import numpy as np
import pytest

from magcore.fem2d.machines.iron_loss import (
    SteinmetzCoefficients,
    hysteresis_density_minor_loops,
    iron_loss_density_from_waveform,
    rainflow_half_ranges,
)
from magcore.fem2d.machines.magnet_loss import (
    effective_eddy_thickness,
    harmonic_rate_spectrum,
    magnet_eddy_loss_density_from_waveform,
    solid_steel_loss_density_from_waveform,
)

# Л-79 (2026-09-11): потери по волне в системе ротора НЕ должны зависеть от выбранного пролёта
# свипа — это числовой параметр. Оракул точный: волна, повторённая k раз на k-кратном пролёте
# (частота повторения отрезка в k раз меньше), описывает ту же физику и обязана дать те же
# потери. До исправления скин-слой кольца и гистерезис брали частоту повторения отрезка, и на
# холостом ходу IM-8008 при 3801 об/мин пролёт 90° давал 41,0 Вт, зубцовый 10° — 27,7 Вт.

SIGMA10 = 1.0 / 0.14e-6
CF = SteinmetzCoefficients.steel10_laminated(0.2e-3)
F_SLOT = 2281.0                     # зубцовая частота IM-8008 при 3801 об/мин
RING = 1.1e-3                       # толщина кольца ротора IM-8008


def _wave(N, harmonics, dc=(0.0, 0.0), n_probes=3, seed=0):
    """Периодическая волна (N,P,2): постоянная часть + гармоники (m, амплитуда) со случайными фазами."""
    rng = np.random.default_rng(seed)
    phi = 2.0 * math.pi * np.arange(N) / N
    B = np.zeros((N, n_probes, 2))
    B[:, :, 0] += dc[0]
    B[:, :, 1] += dc[1]
    for m, amp in harmonics:
        for p in range(n_probes):
            for c in range(2):
                a = amp * rng.uniform(0.5, 1.0)
                B[:, p, c] += a * np.cos(m * phi + rng.uniform(0.0, 2.0 * math.pi))
    return B


def _ring(B, freq):
    idx = np.arange(B.shape[1])
    return solid_steel_loss_density_from_waveform(
        B, idx, idx.size, freq=freq, sigma=SIGMA10, thickness=RING, mu_r=500.0, coeffs=CF)


def _ring_eddy(B, freq):
    idx = np.arange(B.shape[1])
    return _ring(B, freq) - hysteresis_density_minor_loops(B, idx, idx.size, freq=freq, coeffs=CF)


# ----------------------------------------------------------------------- «дождевой поток»
def test_rainflow_known_sequence():
    # 3 → 0 → 2 → 1 → (3): большая петля 0…3 и малая 1…2
    full, half = rainflow_half_ranges([3.0, 0.0, 2.0, 1.0])
    assert np.allclose(np.sort(full), [0.5, 1.5])
    assert half.size == 0


def test_rainflow_does_not_depend_on_start_point():
    x = np.array([3.0, 0.0, 2.0, 1.0, 2.5, 0.5])
    ref = np.sort(rainflow_half_ranges(x)[0])
    for s in range(1, x.size):
        assert np.allclose(np.sort(rainflow_half_ranges(np.roll(x, s))[0]), ref)


def test_rainflow_tiling_multiplies_loops():
    x = _wave(40, [(1, 1.0), (3, 0.3), (7, 0.1)], n_probes=1)[:, 0, 0]
    one = np.sort(rainflow_half_ranges(x)[0])
    four = np.sort(rainflow_half_ranges(np.tile(x, 4))[0])
    assert np.allclose(four, np.sort(np.tile(one, 4)))


def test_rainflow_constant_signal_has_no_loops():
    full, half = rainflow_half_ranges(np.full(10, 1.5))
    assert full.size == 0 and half.size == 0


# ----------------------------------------------------------------------- гистерезис по малым петлям
def test_minor_loops_alternating_field_equals_classic_formula():
    # переменное поле одного направления (30°): одна петля с полуразмахом B_m — ровно k_h·f·B_m^α
    N, Bm, f = 64, 0.8, 500.0
    phi = 2.0 * math.pi * np.arange(N) / N
    B = np.zeros((N, 1, 2))
    B[:, 0, 0] = Bm * np.sin(phi) * math.cos(math.radians(30.0))
    B[:, 0, 1] = Bm * np.sin(phi) * math.sin(math.radians(30.0))
    q = hysteresis_density_minor_loops(B, np.array([0]), 1, freq=f, coeffs=CF)
    classic = CF.k_hyst * f * Bm ** CF.alpha * CF.density
    assert abs(q[0] - classic) / classic < 1e-9


def test_minor_loops_ignore_the_constant_flux():
    # кольцо ротора: постоянный поток 1,5 Тл + зубцовая пульсация 0,05 Тл, 9 периодов на отрезке.
    # Петель девять, у каждой полуразмах 0,05; постоянная часть петли не образует.
    N, f_span = 72, 253.0
    phi = 2.0 * math.pi * np.arange(N) / N
    B = np.zeros((N, 1, 2))
    B[:, 0, 0] = 1.5 + 0.05 * np.cos(9 * phi)
    q = hysteresis_density_minor_loops(B, np.array([0]), 1, freq=f_span, coeffs=CF)
    expected = CF.k_hyst * (9 * f_span) * 0.05 ** CF.alpha * CF.density
    assert abs(q[0] - expected) / expected < 1e-9
    # прежняя формула «пик |B| × частота отрезка» считала постоянный поток петлёй
    peak = iron_loss_density_from_waveform(B, np.array([0]), 1, freq=f_span, coeffs=CF,
                                           hysteresis="peak")
    loops = iron_loss_density_from_waveform(B, np.array([0]), 1, freq=f_span, coeffs=CF,
                                            hysteresis="loops")
    eddy = loops[0] - q[0]
    assert peak[0] - eddy > 20.0 * q[0]


def test_minor_loops_rotating_field_counts_both_axes():
    # вращающееся поле постоянного модуля: приближение «эллипса» — две петли (по двум осям)
    N, Bm, f = 64, 1.0, 400.0
    phi = 2.0 * math.pi * np.arange(N) / N
    B = np.stack([Bm * np.cos(phi), Bm * np.sin(phi)], axis=1)[:, None, :]
    q = hysteresis_density_minor_loops(B, np.array([0]), 1, freq=f, coeffs=CF)
    two = 2.0 * CF.k_hyst * f * Bm ** CF.alpha * CF.density
    assert abs(q[0] - two) / two < 0.01


def test_minor_loops_zero_frequency_and_constant_field():
    B = _wave(16, [(1, 0.1)], dc=(1.0, 0.0))
    idx = np.arange(B.shape[1])
    assert np.all(hysteresis_density_minor_loops(B, idx, idx.size, freq=0.0, coeffs=CF) == 0.0)
    const = np.full((16, 2, 2), 1.3)
    assert np.all(hysteresis_density_minor_loops(const, np.arange(2), 2, freq=500.0, coeffs=CF) == 0.0)


# ----------------------------------------------------------------------- независимость от пролёта
def test_ring_loss_does_not_depend_on_span():
    # волна кольца: постоянный поток + три зубцовые гармоники; тот же сигнал на отрезке из девяти
    # периодов (как пролёт 90° против зубцового 10° у 36/40) обязан дать те же потери
    B = _wave(8, [(1, 0.08), (2, 0.03), (3, 0.01)], dc=(1.4, 0.2))
    q1 = _ring(B, F_SLOT)
    q9 = _ring(np.tile(B, (9, 1, 1)), F_SLOT / 9.0)
    assert np.all(q1 > 0.0)
    assert np.allclose(q9, q1, rtol=1e-9, atol=0.0)


def test_laminated_rotor_iron_with_loops_does_not_depend_on_span():
    B = _wave(8, [(1, 0.08), (3, 0.02)], dc=(1.2, -0.3))
    idx = np.arange(B.shape[1])
    kw = dict(coeffs=CF, hysteresis="loops")
    q1 = iron_loss_density_from_waveform(B, idx, idx.size, freq=F_SLOT, **kw)
    q9 = iron_loss_density_from_waveform(np.tile(B, (9, 1, 1)), idx, idx.size, freq=F_SLOT / 9.0, **kw)
    assert np.allclose(q9, q1, rtol=1e-9, atol=0.0)


def test_peak_hysteresis_depends_on_span_hence_stator_only():
    # «пик × частота отрезка» годится только для волны за ОДИН период переменного поля (статор):
    # на отрезке из девяти периодов она ошибается в разы — поэтому для ротора запрещена
    B = _wave(8, [(1, 0.08)], dc=(1.2, 0.0))
    idx = np.arange(B.shape[1])
    kw = dict(coeffs=CF, hysteresis="peak")
    q1 = iron_loss_density_from_waveform(B, idx, idx.size, freq=F_SLOT, **kw)
    q9 = iron_loss_density_from_waveform(np.tile(B, (9, 1, 1)), idx, idx.size, freq=F_SLOT / 9.0, **kw)
    assert np.all(q1 > 3.0 * q9)


def test_unknown_hysteresis_model_rejected():
    B = _wave(8, [(1, 0.08)])
    idx = np.arange(B.shape[1])
    with pytest.raises(ValueError):
        iron_loss_density_from_waveform(B, idx, idx.size, freq=F_SLOT, coeffs=CF, hysteresis="max")


# ----------------------------------------------------------------------- совместимость и гармоники
@pytest.mark.parametrize("N", [24, 25])
def test_harmonic_rate_spectrum_sums_to_central_difference(N):
    B = _wave(N, [(1, 0.1), (2, 0.05), (5, 0.02), (11, 0.01)], dc=(0.7, 0.1))
    idx = np.arange(B.shape[1])
    f_m, S = harmonic_rate_spectrum(B, 300.0)
    q_ref = magnet_eddy_loss_density_from_waveform(B, idx, idx.size, freq=300.0, sigma=1.0, seg_width=1.0)
    assert np.allclose(S.sum(axis=0) / 12.0, q_ref, rtol=1e-10, atol=0.0)
    assert np.allclose(f_m, 300.0 * np.arange(f_m.size))


def test_single_harmonic_eddy_equals_previous_formula():
    # одна гармоника на отрезке — случай, где прежняя формула была верна: σ·w_eff(f)²/12·⟨(dB/dt)²⟩
    f = 1400.0
    B = _wave(64, [(1, 0.05)])
    idx = np.arange(B.shape[1])
    w_eff = effective_eddy_thickness(RING, f, SIGMA10, 500.0)
    old = magnet_eddy_loss_density_from_waveform(B, idx, idx.size, freq=f, sigma=SIGMA10, seg_width=w_eff)
    assert np.allclose(_ring_eddy(B, f), old, rtol=1e-9, atol=0.0)


def test_each_harmonic_gets_its_own_skin_depth():
    # две гармоники (m = 1 и m = 4): вихревые = сумма потерь каждой, у четвёртой скин-слой тоньше
    f = 1400.0
    B1 = _wave(64, [(1, 0.05)], n_probes=1, seed=1)
    B4 = _wave(64, [(4, 0.02)], n_probes=1, seed=2)
    both = _ring_eddy(B1 + B4, f)[0]
    assert abs(both - (_ring_eddy(B1, f)[0] + _ring_eddy(B4, f)[0])) / both < 1e-9
    w4 = effective_eddy_thickness(RING, 4.0 * f, SIGMA10, 500.0)
    old4 = magnet_eddy_loss_density_from_waveform(B4, np.array([0]), 1, freq=f, sigma=SIGMA10, seg_width=w4)
    assert abs(_ring_eddy(B4, f)[0] - old4[0]) / old4[0] < 1e-9
    assert w4 < effective_eddy_thickness(RING, f, SIGMA10, 500.0)
