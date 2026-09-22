from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.excitation import winding_current_density
from magcore.fem2d.machines.iron_loss import (
    SteinmetzCoefficients,
    hysteresis_density_minor_loops,
    iron_loss_density_from_waveform,
)
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.rotor_sweep import _solve_at_angle, sample_B_at_points
from magcore.fem2d.machines.winding import star_of_slots_layout

# P-B4: вихревые в МАГНИТЕ + потери в стали РОТОРА — источники тепла, греющие магнит НАПРЯМУЮ.
#
# Ключ: и то и другое наводится ПУЛЬСАЦИЯМИ поля в СИСТЕМЕ РОТОРА (зубцовые + якорные гармоники),
# а НЕ фундаменталом. В лабораторной системе поле магнита огромно и «вращается» → даёт фантомный
# dB/dt; в системе ротора собственное поле магнита ПОСТОЯННО (едет вместе с ним), и остаётся
# только пульсация. Поэтому B снимается в СО-ВРАЩАЮЩИХСЯ точках (лаб = Rot(a)·p_ротор) и вектор
# поворачивается обратно в систему ротора (B_ротор = Rot(−a)·B_лаб).
#
# Вихревые в проводящей пластине шириной w в поле B(t): P_об = (σ·w²/12)·⟨(dB/dt)²⟩ [Вт/м³];
# для синусоиды B_m это σ·w²·(2πf)²·B_m²/24. Сегментация: w→w/N_seg ⇒ P ∝ 1/N_seg².
# Через безразмерную фазу φ∈[0,2π) снятого пролёта: dB/dt = (2π·f_ref)·dB/dφ,
# f_ref = ω_мех/mech_span (для полного оборота = n/60; для зубцового шага = n_slots·n/60).
#
# ГРАНИЦА (честно): скалярная 1D-оценка вихревых (skin depth ≫ w); w — представительная ширина
# сегмента. Полный оборот (mech_span=2π) захватывает ВСЕ гармоники (зубцовые+якорные), но требует
# N≳48 (12 зубцовых пролётов/об); зубцовый шаг (mech_span=2π/n_slots) дёшев и ловит доминирующую
# зубцовую пульсацию, но не якорные/ШИМ-гармоники (последние — через множитель, ШИМ×2, §12).
#
# НЕЗАВИСИМОСТЬ ОТ ПРОЛЁТА (Л-79, 2026-09-11). Пролёт свипа — ЧИСЛОВОЙ параметр: при любом
# пролёте, кратном периоду поля, ответ обязан быть один. Вихревые магнита таковы сами собой:
# (2π·f_ref)²·⟨(dB/dφ)²⟩ = ⟨(dB/dt)²⟩. Члены, зависящие от ЧАСТОТЫ, — скин-слой кольца и
# гистерезис — берут частоту КАЖДОЙ гармоники (m·f_ref) и считают каждую петлю, а не f_ref
# отрезка. Раньше брали f_ref: на холостом ходу IM-8008 при 3801 об/мин пролёт 90° давал
# 41,0 Вт, зубцовый 10° — 27,7 Вт при одной и той же волне. Закреплено тестами
# `test_rotor_loss_span_invariance.py`: волна, повторённая k раз на k-кратном пролёте, даёт
# те же потери.


def skin_depth(freq: float, sigma: float, mu_r: float) -> float:
    """Глубина проникновения δ=√(2/(ω·μ·σ)) [м] (ω=2πf, μ=μ_r·μ₀)."""
    omega = 2.0 * math.pi * float(freq)
    if omega <= 0.0 or sigma <= 0.0 or mu_r <= 0.0:
        return float("inf")
    return math.sqrt(2.0 / (omega * float(mu_r) * MU0 * float(sigma)))


def effective_eddy_thickness(thickness: float, freq: float, sigma: float, mu_r: float) -> float:
    """
    Эффективная толщина для вихревых потерь с учётом ВЫТЕСНЕНИЯ ТОКА (скин-эффект).

    При δ ≫ w поле проникает насквозь ⇒ классический режим (потери ∝ σw²).
    При δ ≪ w ток вытеснен в скин-слой ⇒ участвует лишь ~δ, и формула ∝ σw² ЗАВЫШАЕТ
    (для массивного ярма Ст10 при 1400 Гц: δ≈0.16 мм против w=3 мм — завышение ~89×).
    Инженерная сшивка режимов: **w_eff = min(w, 2δ)**.

    ⚠ ГРАНИЦА МОДЕЛИ: сшивка приближённая (точное решение — через поверхностный импеданс),
    и δ зависит от μ_r, которая в насыщающейся стали меняется ⇒ параметр входит в анализ
    чувствительности (§12). Для магнита (μ_r≈1.05) δ≈12–15 мм > ширины полюса ⇒ предел не
    срабатывает, и модель P-B4 остаётся классической.
    """
    return min(float(thickness), 2.0 * skin_depth(freq, sigma, mu_r))


def _rotate(v: np.ndarray, angle: float) -> np.ndarray:
    """Повернуть набор 2D-векторов/точек на +angle (Rot(angle)·v)."""
    c, s = math.cos(float(angle)), math.sin(float(angle))
    v = np.asarray(v, dtype=float)
    return np.column_stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c])


def magnet_segment_width(params: OutrunnerPMSMParams, n_seg: int = 1) -> float:
    """
    Представительная ширина сегмента магнита [м] = дуговая ширина полюса / N_seg.
    Дуга полюса = embrace·(2π/n_poles)·R_mag_ср. Сегментация делит её на N_seg.
    """
    r_mag = 0.5 * (params.R_mag_in + params.R_mag_out)
    pole_arc = params.magnet_embrace * (2.0 * math.pi / params.n_poles) * r_mag
    return float(pole_arc / max(int(n_seg), 1))


def magnet_eddy_loss_density_from_waveform(
    B_series: np.ndarray, cell_idx: np.ndarray, n_cells: int, *,
    freq: float, sigma: float, seg_width: float,
) -> np.ndarray:
    """
    Плотность вихревых потерь магнита q_pm [Вт/м³] по ячейкам из волны B_ротор(φ) за пролёт.

    `B_series` (N,P,2) — волна в системе РОТОРА за один снятый пролёт (φ=0…2π в N точках);
    `freq` — f_ref пролёта. P_об = (σ·w²/12)·(2π·f)²·⟨(dB/dφ)²⟩ (для синусоиды = σw²(2πf)²B²/24).
    Чистая функция — проверяется синтетической волной без прогонки ротора.
    """
    B = np.asarray(B_series, dtype=float)
    idx = np.asarray(cell_idx, dtype=int).reshape(-1)
    if B.ndim != 3 or B.shape[1] != idx.size or B.shape[2] != 2:
        raise ValueError("B_series должно быть (N, P, 2) с P = число cell_idx.")
    N = B.shape[0]
    dphi = 2.0 * math.pi / N
    dB = (np.roll(B, -1, axis=0) - np.roll(B, 1, axis=0)) / (2.0 * dphi)   # периодич. производная
    mean_sq = (dB ** 2).sum(axis=2).mean(axis=0)                          # ⟨(dB/dφ)²⟩ по ячейке
    p_vol = (float(sigma) * float(seg_width) ** 2 / 12.0) * (2.0 * math.pi * float(freq)) ** 2 * mean_sq
    q = np.zeros(int(n_cells), dtype=float)
    q[idx] = p_vol
    return q


def harmonic_rate_spectrum(B_series: np.ndarray, freq: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Разложение ⟨(dB/dt)²⟩ волны B (N,P,2) по ГАРМОНИКАМ снятого отрезка.

    Возвращает (f_m, S): f_m = m·freq [Гц] — частоты гармоник m = 0…⌊N/2⌋, S (M,P) — вклад
    гармоники m в ⟨(dB/dt)²⟩ по пробной точке (сумма по компонентам). Вес гармоники — тот же,
    что даёт периодическая центральная разность (sin(mΔφ)/Δφ вместо m), поэтому Σ_m S в точности
    равна среднему, по которому считает `magnet_eddy_loss_density_from_waveform`: разложение
    ничего не меняет, пока множитель от частоты не зависит, и позволяет взять частотно-зависимый
    множитель (скин-слой) у КАЖДОЙ гармоники.
    """
    B = np.asarray(B_series, dtype=float)
    if B.ndim != 3 or B.shape[2] != 2:
        raise ValueError("B_series должно быть (N, P, 2).")
    N = B.shape[0]
    X = np.fft.rfft(B, axis=0)                                   # (M, P, 2)
    m = np.arange(X.shape[0])
    dphi = 2.0 * math.pi / N
    g = np.sin(m * dphi) / dphi                                  # множитель центральной разности
    g[0] = 0.0
    if N % 2 == 0:
        g[-1] = 0.0                                              # Найквист: разность его не видит
    power = (np.abs(X) ** 2).sum(axis=2) * (2.0 / N ** 2)
    S = (2.0 * math.pi * float(freq)) ** 2 * (g ** 2)[:, None] * power
    return m * float(freq), S


def solid_steel_loss_density_from_waveform(
    B_series: np.ndarray, cell_idx: np.ndarray, n_cells: int, *,
    freq: float, sigma: float, thickness: float, mu_r: float,
    coeffs: SteinmetzCoefficients,
) -> np.ndarray:
    """
    Плотность потерь в МАССИВНОЙ (нешихтованной) стали [Вт/м³] по волне B (N,P,2) за снятый
    отрезок, повторяющийся с частотой `freq`.

    Вихревые — по модели сплошного тела со СКИН-ПРЕДЕЛОМ, ПО ГАРМОНИКАМ отрезка: у гармоники m
    своя частота f_m = m·freq и своя эффективная толщина w_eff(f_m) = min(w, 2δ(f_m)):
        q_e = (σ/12)·Σ_m w_eff(f_m)²·S_m,   S_m — вклад гармоники в ⟨(dB/dt)²⟩
    (`harmonic_rate_spectrum`). Гистерезис — по малым петлям (`hysteresis_density_minor_loops`):
    постоянный поток кольца петлю не образует.

    ⚠ ИСПРАВЛЕНО 2026-09-11 (Л-79): раньше и δ, и гистерезис брали частоту повторения отрезка
      `freq` — ответ зависел от выбора пролёта свипа.
    ⚠ Удельные ВИХРЕВЫЕ Вт/кг Штейнмеца здесь НЕПРИМЕНИМЫ: они калиброваны на толщину ЛИСТА
      шихтовки. Гистерезисный член от толщины не зависит и переиспользуется как есть.
    ⚠ ГРАНИЦА МОДЕЛИ: сшивка min(w, 2δ) и замыкание вихревого тока поперёк толщины в каждой
      ячейке — приближения; формулу выбирать по эталонному расчёту вихревых токов.
    """
    B = np.asarray(B_series, dtype=float)
    idx = np.asarray(cell_idx, dtype=int).reshape(-1)
    if B.ndim != 3 or B.shape[1] != idx.size or B.shape[2] != 2:
        raise ValueError("B_series должно быть (N, P, 2) с P = число cell_idx.")
    f_m, S = harmonic_rate_spectrum(B, freq)
    w_eff = np.array([effective_eddy_thickness(thickness, f, sigma, mu_r) for f in f_m])
    q = np.zeros(int(n_cells), dtype=float)
    q[idx] = (float(sigma) / 12.0) * (w_eff[:, None] ** 2 * S).sum(axis=0)
    q += hysteresis_density_minor_loops(B, idx, n_cells, freq=freq, coeffs=coeffs)
    return q


def rotor_frame_B_series(
    params: OutrunnerPMSMParams, magnet: AnisotropicBHTMagnet, steel: SteelBHCurve, *,
    speed_rpm: float, cell_idx: np.ndarray, i_peak: float = 0.0, gamma_elec: float = 0.0,
    turns_per_slot: float = 0.0, T: float = 20.0, mech_span: float | None = None,
    n_positions: int = 48, relaxation: float = 0.1, max_iter: int = 300, tol: float = 1.0e-6,
) -> tuple[np.ndarray, float]:
    """
    Волна B в СИСТЕМЕ РОТОРА в заданных ячейках за механический пролёт `mech_span`.

    `cell_idx` — ячейки эталонной (угол 0) геометрии = роторные опорные точки. На каждом угле a
    геометрия перестраивается, поле решается, B снимается в лаб-точках Rot(a)·p_ротор и
    поворачивается в систему ротора Rot(−a)·B_лаб. Возвращает (B_series (N,P,2), f_ref).
    mech_span по умолчанию 2π (полный оборот, все гармоники). f_ref = (2π·n/60)/mech_span.
    """
    span = 2.0 * math.pi if mech_span is None else float(mech_span)
    geo0 = build_outrunner_spm_pmsm(params)
    idx = np.asarray(cell_idx, dtype=int).reshape(-1)
    p_rotor = np.array([geo0.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
    p_pairs = params.n_poles // 2
    layout = star_of_slots_layout(params.n_slots, params.n_poles)
    have_current = i_peak != 0.0 and turns_per_slot != 0.0

    angles = np.arange(int(n_positions)) * span / int(n_positions)
    B_series = np.empty((int(n_positions), idx.size, 2), dtype=float)
    for i, a in enumerate(angles):
        geo = build_outrunner_spm_pmsm(replace(params, rotor_angle=float(a)))
        gamma_abs = float(gamma_elec) + p_pairs * float(a)
        jz = (winding_current_density(geo, layout, i_peak=i_peak, gamma_elec=gamma_abs,
                                      turns_per_slot=turns_per_slot) if have_current else None)
        em = _solve_at_angle(geo, magnet, steel, T=T, retention=None, j_cells=jz,
                             relaxation=relaxation, max_iter=max_iter, tol=tol)
        lab_pts = _rotate(p_rotor, a)                        # ротор → лаб
        B_lab = sample_B_at_points(geo.mesh, em.B_cells, lab_pts)
        B_series[i] = _rotate(B_lab, -a)                     # лаб → ротор (вектор)

    f_ref = (2.0 * math.pi * float(speed_rpm) / 60.0) / span
    return B_series, f_ref


def magnet_rotor_loss_density(
    params: OutrunnerPMSMParams, magnet: AnisotropicBHTMagnet, steel: SteelBHCurve, *,
    speed_rpm: float, sigma_pm: float, magnet_seg_width: float,
    i_peak: float = 0.0, gamma_elec: float = 0.0, turns_per_slot: float = 0.0, T: float = 20.0,
    steinmetz: SteinmetzCoefficients | None = None, mech_span: float | None = None,
    n_positions: int = 48, relaxation: float = 0.1, max_iter: int = 300,
    rotor_solid: bool = False, sigma_rotor: float | None = None,
    rotor_thickness: float | None = None, rotor_mu_r: float = 500.0,
) -> tuple[np.ndarray, float]:
    """
    Поэлементная карта потерь РОТОР-СТОРОНЫ q [Вт/м³] (n_cells,) = вихревые магнита + сталь ротора
    (оба из поля в системе ротора, греют магнит). ОДИН свип на обе области. Возвращает (q, f_ref).

    `rotor_solid=True` — ярмо ротора **массивное** (не шихтованное, напр. Сталь 10 изделия):
    вихревые считаются по модели сплошного тела со скин-пределом (`sigma_rotor`,
    `rotor_thickness` по умолчанию = радиальная толщина ярма, `rotor_mu_r`), а не по
    удельным Вт/кг Штейнмеца. Гистерезис — в обоих случаях по малым петлям (Л-79).
    Свип и потери разделены: `rotor_frame_B_series` + `rotor_side_loss_density_from_waveform`.
    """
    geo0 = build_outrunner_spm_pmsm(params)
    mag_idx, ry_idx = rotor_side_cells(geo0)
    B, f_ref = rotor_frame_B_series(
        params, magnet, steel, speed_rpm=speed_rpm, cell_idx=np.concatenate([mag_idx, ry_idx]),
        i_peak=i_peak, gamma_elec=gamma_elec, turns_per_slot=turns_per_slot, T=T,
        mech_span=mech_span, n_positions=n_positions, relaxation=relaxation, max_iter=max_iter,
    )
    q = rotor_side_loss_density_from_waveform(
        geo0, B, f_ref, sigma_pm=sigma_pm, magnet_seg_width=magnet_seg_width, steinmetz=steinmetz,
        rotor_solid=rotor_solid, sigma_rotor=sigma_rotor, rotor_thickness=rotor_thickness,
        rotor_mu_r=rotor_mu_r)
    return q, f_ref


def rotor_side_cells(geometry) -> tuple[np.ndarray, np.ndarray]:
    """Ячейки ротор-стороны эталонной геометрии: (магнит, ярмо ротора) — в этом порядке идёт волна."""
    mag_idx = np.where(geometry.region == int(Region.MAGNET))[0]
    ry_idx = np.where(geometry.region == int(Region.ROTOR_YOKE))[0]
    return mag_idx, ry_idx


def rotor_side_loss_density_from_waveform(
    geometry, B_series: np.ndarray, f_ref: float, *, sigma_pm: float, magnet_seg_width: float,
    steinmetz: SteinmetzCoefficients | None = None, rotor_solid: bool = False,
    sigma_rotor: float | None = None, rotor_thickness: float | None = None,
    rotor_mu_r: float = 500.0,
) -> np.ndarray:
    """
    Карта потерь ротор-стороны q [Вт/м³] (n_cells,) по УЖЕ СНЯТОЙ волне B (N,P,2) в ячейках
    `rotor_side_cells(geometry)` (сначала магнит, затем ярмо) за отрезок с частотой повторения
    `f_ref`. Отделено от свипа, чтобы дорогую волну можно было хранить, а формулы — менять
    без новых свипов.
    """
    mag_idx, ry_idx = rotor_side_cells(geometry)
    B = np.asarray(B_series, dtype=float)
    if B.ndim != 3 or B.shape[1] != mag_idx.size + ry_idx.size or B.shape[2] != 2:
        raise ValueError("волна должна быть (N, P, 2) по ячейкам rotor_side_cells(geometry).")
    nc = geometry.mesh.n_cells
    nm = mag_idx.size
    cf = steinmetz or SteinmetzCoefficients.m270_35a_cogent()
    q = magnet_eddy_loss_density_from_waveform(
        B[:, :nm], mag_idx, nc, freq=f_ref, sigma=sigma_pm, seg_width=magnet_seg_width)
    if rotor_solid:
        if sigma_rotor is None:
            raise ValueError("rotor_solid=True требует sigma_rotor (проводимость стали ротора).")
        w = geometry.params.h_rotor_yoke if rotor_thickness is None else float(rotor_thickness)
        q += solid_steel_loss_density_from_waveform(
            B[:, nm:], ry_idx, nc, freq=f_ref, sigma=sigma_rotor, thickness=w,
            mu_r=rotor_mu_r, coeffs=cf)
    else:
        q += iron_loss_density_from_waveform(B[:, nm:], ry_idx, nc, freq=f_ref, coeffs=cf,
                                             hysteresis="loops")
    return q
