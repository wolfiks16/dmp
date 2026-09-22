import numpy as np

from magcore.constants import MU0
from magcore.domain.steel_curves import (
    m270_35a_bh_curve,
    m270_35a_cogent_bh_curve,
    steel10_bh_curve,
)

# Сталь 10 (данные изделия). Оракулы расшифровки СГС→СИ и физической состоятельности:
# насыщение, начальная проницаемость, монотонность, dH/dB≤1/μ0, сопоставление с M270.


def test_steel10_decoding_physical():
    c = steel10_bh_curve()
    B, H = c.B_values, c.H_values
    # диапазоны из расшифровки (B[Гс]/1e4, H[Э]·79.577)
    assert abs(B.max() - 2.2458) < 1e-6            # насыщение 2.246 Тл
    assert abs(H.max() - 1905.0 * 79.5774715) < 1e-3
    assert B[0] == 0.0 and H[0] == 0.0
    assert np.all(np.diff(B) > 0) and np.all(np.diff(H) > 0)
    # начальная относительная проницаемость низкоуглеродистой стали ~2000-3500
    mu_r0 = B[1] / (MU0 * H[1])
    assert 2000.0 < mu_r0 < 3500.0
    # дифф. проницаемость не ниже вакуума (валидатор класса это и требует)
    assert np.max(np.diff(H) / np.diff(B)) <= 1.0 / MU0


def test_steel10_matches_cogent_datasheet_hardness():
    # Сталь 10 в рабочей точке 1.5 Тл требует 1681 А/м — практически как РЕАЛЬНЫЙ datasheet
    # M270-35A (Cogent: 1700 А/м). ⚠ Представительная кривая M270 в репозитории даёт лишь
    # ~700 А/м, т.е. она заметно МЯГЧЕ реального листа (см. sources_registry.md §1).
    s10 = steel10_bh_curve()
    h10 = np.interp(1.5, s10.B_values, s10.H_values)
    assert abs(h10 - 1700.0) / 1700.0 < 0.10          # согласие с datasheet-жёсткостью
    h_repr = np.interp(1.5, m270_35a_bh_curve().B_values, m270_35a_bh_curve().H_values)
    assert h_repr < h10                                 # представительная M270 мягче — зафиксировано
    # выше по индукции сталь 10 несёт больше потока (насыщение 2.25 против ≈2.2)
    assert s10.B_values.max() > m270_35a_bh_curve().B_values.max()


def test_m270_cogent_datasheet_curve():
    # Datasheet-кривая Cogent: 1.5 Тл при 1700 А/м (реальный лист), в отличие от
    # «представительной» (700 А/м). Она — корректная база сравнения со Сталью 10.
    c = m270_35a_cogent_bh_curve()
    assert abs(np.interp(1.5, c.B_values, c.H_values) - 1700.0) < 1.0
    assert np.all(np.diff(c.B_values) > 0) and np.all(np.diff(c.H_values) > 0)
    assert np.max(np.diff(c.H_values) / np.diff(c.B_values)) <= 1.0 / MU0
    # жёсткость совпала со Сталью 10 в рабочей точке (стали расходятся ПОТЕРЯМИ, не B-H)
    h10 = np.interp(1.5, steel10_bh_curve().B_values, steel10_bh_curve().H_values)
    assert abs(np.interp(1.5, c.B_values, c.H_values) - h10) / h10 < 0.05
    # и в 2+ раза жёстче прежней представительной
    assert np.interp(1.5, c.B_values, c.H_values) > 2.0 * np.interp(
        1.5, m270_35a_bh_curve().B_values, m270_35a_bh_curve().H_values)


def test_curves_reach_saturation_without_permeability_cliff():
    # ЗАЩИТА ОТ РЕАЛЬНОЙ ОШИБКИ: если таблица обрывается там, где сталь ещё магнитно активна,
    # класс экстраполирует наклоном 1/μ0 ⇒ СКАЧОК проницаемости на границе, и нелинейный решатель
    # разваливается в зубцах (проверено: карта на обрезанной на 1.8 Тл кривой не сделала ни шага).
    # Требование: последний сегмент таблицы уже близок к вакууму (μ_r ≲ 2) — т.е. насыщение достигнуто.
    for c in (steel10_bh_curve(), m270_35a_cogent_bh_curve()):
        mu_r_last = 1.0 / ((np.diff(c.H_values)[-1] / np.diff(c.B_values)[-1]) * MU0)
        assert mu_r_last < 2.0, f"{c.name}: таблица обрывается при mu_r={mu_r_last:.1f} (обрыв)"
        assert c.B_values.max() >= 2.0, f"{c.name}: таблица не доходит до насыщения"


def test_steel10_reluctivity_interface():
    c = steel10_bh_curve()
    # хордовая/дифференциальная ν пригодны для решателя (конечны, положительны)
    for b in (0.5, 1.0, 1.5, 2.0, 2.5):
        nu, nu_d = c.nu_pair(b)
        assert np.isfinite(nu) and nu > 0.0
        assert np.isfinite(nu_d) and nu_d > 0.0
    # глубокое насыщение → ν стремится к 1/μ0 сверху не выходя
    assert c.nu_chord(5.0) <= 1.0 / MU0 * 1.01
    assert abs(c.nu_saturation - 1.0 / MU0) / (1.0 / MU0) < 1e-9
