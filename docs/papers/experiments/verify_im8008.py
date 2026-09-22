# -*- coding: utf-8 -*-
"""
НЕЗАВИСИМАЯ СВЕРКА модели Scorpion IM-8008-100kv с отчётом об испытаниях — ПРОХОДЫ 2–3.

ЧТО БЫЛО В ПРОХОДЕ 1 (2026-09-10) и почему пересчитываем:
  · Этап А считал ток шины как (3/π)·I_фазы. Это соотношение формы тока при ПОЛНОЙ
    модуляции; при неполном газе регулятор понижает напряжение и повышает ток, и формула
    завышала ток втрое. Совпадение «−0,7 %» при полном газе было взаимным гашением двух
    ошибок: глубина модуляции даже при газе 2000 мкс — 0,87, а не 1. ЗАМЕНЕНО балансом
    мощности.
  · Этап Б дал невязку КПД от −6,5 до +9,2 п.п., монотонно по частоте. Диагноз:
      – на 439 Гц модель давала потерь в ядре 15,1 Вт при ВСЕХ измеренных 10,9 Вт — это
        невозможно: статор считался из Стали 10, а серийный мотор — на кремнистой стали;
      – на высоких частотах недоставало потерь: в модели меди не было ни лобовых частей,
        ни переменной составляющей.

ПРОХОД 2 — АБЛЯЦИЯ, по одной правке за шаг:
  шаг 0  как в проходе 1: статор Сталь 10, медь по активной длине, постоянный ток;
  шаг 1  статор — кремнистая сталь 0,2 мм (ротор остаётся массивной сталью — так и есть);
  шаг 2  + лобовые части в сопротивлении;
  шаг 3  + переменная составляющая в пазу (Дауэлл) — ДИАПАЗОНОМ по возможной намотке.
  Итог: противоречие на малых оборотах ушло, тренд уменьшился вдвое, но на большой нагрузке
  остался остаток до 116 Вт ∝ f⁵; из проверенных механизмов лучше всего его описывают
  вихревые токи от гармоник реакции якоря (∝ I²·f²).

ПРОХОД 3 — ШАГ 4: ПОТЕРИ В ЯДРЕ ПОД НАГРУЗКОЙ.
  Причина остатка оказалась в самой сверке: все свипы потерь шли при НУЛЕВОМ токе, то есть
  учитывали только зубцовые гармоники холостого хода. Под нагрузкой дробная обмотка 36N40P
  создаёт пространственные гармоники МДС, вращающиеся несинхронно с ротором; они наводят
  вихревые токи в магнитах и в сплошном кольце ротора. Теперь потери ядра считаются в
  ЧЕТЫРЁХ точках самой винтовой характеристики, каждая при своём фазном токе M/K_t и при
  угле тока максимального момента, и интерполируются по частоте в двойном логарифмическом
  масштабе (форма зависимости не навязывается).
  ⚠ Пролёт свипа ротора — ПЕРИОД ОБМОТКИ 2π/НОД(36, 40) = 90°, а не зубцовое деление:
    под нагрузкой поле в системе ротора повторяется только через четверть оборота; на
    зубцовом делении оно НЕ периодично, и производная на стыке дала бы ложный скачок.
    72 положения = 8 на зубцовое деление, как в холостом свипе, — иначе центральная разность
    занизила бы dB/dt для зубцовой гармоники.
  ⚠ ПРЕДСКАЗАНИЕ ЗАФИКСИРОВАНО ДО ПРОГОНА (константы PREDICTION_*): если остаток — это
    вихревые от реакции якоря, подразумеваемый КПД регулятора станет ровным в пределах
    0,94…0,99 во всех годных точках, а тренд невязки по модулю < 2 п.п./кГц.

ПРОХОД 4 (2026-09-11) — ПОТЕРИ РОТОРА БЕЗ ЗАВИСИМОСТИ ОТ ПРОЛЁТА СВИПА (Л-79).
  Контроль прохода 3 не сошёлся: ротор на холостом ходу при 3801 об/мин — пролёт 90° дал
  41,0 Вт, зубцовый 10° — 27,7 Вт при одной и той же волне. Причина — в формуле кольца ротора:
  глубина проникновения и гистерезис брали частоту повторения снятого отрезка, а не частоты
  гармоник поля. Исправлено в magnet_loss.py / iron_loss.py: вихревые кольца — по гармоникам,
  гистерезис — по малым петлям. Теперь:
  · кэшируются ВОЛНЫ B ротора (waves/), а не числа потерь: формулы считаются заново при каждом
    запуске, и устаревшее число из кэша взять нельзя (Л-80);
  · потери ротора на холостом ходу — прямо из волны при каждой скорости, без подгонки k1·f + k2·f²;
  · КОНТРОЛЬ: пролёт 90° против девяти зубцовых окон 10° той же волны холостого хода; критерий
    задан до прогона — значение на 90° лежит внутри разброса девяти окон.
  Критерии предсказания шага 4 НЕ меняются.

ЧЕСТНЫЕ ОГОВОРКИ:
  1. Измеренный КПД — мотор ВМЕСТЕ С регулятором. Печатается «подразумеваемый КПД
     регулятора» = КПД_изм / КПД_модели; > 1 — ОТКАЗ модели в данной точке.
  2. Механических потерь (подшипники, вентиляция колокола) в модели НЕТ.
  3. По винтовой характеристике ток и частота жёстко связаны (I ∝ n²) — потери, зависящие
     от тока и от частоты, по этим данным неразделимы в принципе.
  4. Самопротиворечивые точки отчёта (|M·ω − P_мех| > 2 % — порог задан ДО сверки)
     печатаются, но в итоги не входят.
  5. Температура обмотки в отчёте не указана — медь при 80 °C; магнит при 20 °C (горячий
     магнит дал бы меньший K_t и больший ток).
  6. Марки магнита и стали не публикуются; приняты N42SH и кремнистая сталь 0,2 мм. Магнит
     считается НЕСЕГМЕНТИРОВАННЫМ, кольцо ротора — СПЛОШНЫМ: оба входа сильнее всего влияют
     на потери под нагрузкой и закрываются только разборкой.

Запуск: PYTHONPATH=<repo> python -u docs/papers/experiments/verify_im8008.py > out.txt
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np                                                        # noqa: E402

import scenario_paper1 as cfg                                             # noqa: E402
from fig_im8008_section import IM8008, KV, SLOT_FILL                      # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet                      # noqa: E402
from magcore.fem2d.losses import dowell_ac_factor, round_wire_delta       # noqa: E402
from magcore.fem2d.machines.characteristics import (                      # noqa: E402
    phase_resistance,
    tooth_coil_end_length,
)
from magcore.fem2d.machines.conventions import ke_from_kv                 # noqa: E402
from magcore.fem2d.machines.excitation import slot_areas                  # noqa: E402
from magcore.fem2d.machines.iron_loss import (                            # noqa: E402
    SteinmetzCoefficients,
    hysteresis_density_minor_loops,
    stator_iron_loss_density,
)
from magcore.fem2d.machines.magnet_loss import (                          # noqa: E402
    magnet_eddy_loss_density_from_waveform,
    magnet_segment_width,
    rotor_frame_B_series,
    rotor_side_cells,
    rotor_side_loss_density_from_waveform,
    solid_steel_loss_density_from_waveform,
)
from magcore.fem2d.machines.scenario import machine_scenario              # noqa: E402

MOTOR_DIR = HERE.parents[1] / "motors" / "scorpion_im8008"
CSV = MOTOR_DIR / "IM-8008-100kv_measured.csv"
CACHE = MOTOR_DIR / "core_loss_cache.json"
WAVES = MOTOR_DIR / "waves"          # волны B ротора (дорогие свипы); потери считаются из них заново

T_CU = 80.0                         # температура обмотки [°C] — в отчёте НЕ указана
LAMINATION = 0.2e-3                 # ПАСПОРТ: «Stator Specifications 0.2mm»
N_SEG = 1                           # сегментация магнита неизвестна; 1 = консервативно
SPEEDS_FIT = (1400.0, 4000.0)       # холостой ход: двух скоростей хватает на два коэффициента
CONSISTENCY_TOL = 0.02              # порог самопротиворечия отчёта — задан ДО сверки

# --- шаг 4: потери ядра под нагрузкой -------------------------------------------------------
WINDING_PERIOD = 2.0 * math.pi / math.gcd(IM8008.n_slots, IM8008.n_poles)   # 90° для 36/40
N_ROTOR_POS = 72                    # 8 положений на зубцовое деление, как в холостом свипе
N_STATOR_POS = 12
SLOT_PITCH = 2.0 * math.pi / IM8008.n_slots                                  # 10° для 36 пазов
OP_RPMS = (1318.0, 2450.0, 3100.0, 3801.0)    # точки винтовой характеристики (края + середина)
# ПРЕДСКАЗАНИЕ, ЗАФИКСИРОВАННОЕ ДО ПРОГОНА ШАГА 4 (2026-09-10):
PREDICTION_ESC = (0.94, 0.99)       # подразумеваемый КПД регулятора во ВСЕХ годных точках
PREDICTION_SLOPE = 2.0              # |тренд невязки КПД| < 2 п.п./кГц

# ⚠ СЕТКА ДЛЯ ПОТЕРЬ — НАМЕРЕННО ГРУБАЯ. Свипы по положению ротора — десятки нелинейных
#   решений; на боевой сетке (47 тыс. ячеек) прогон не укладывался и в 50 минут. Потери —
#   ИНТЕГРАЛЬНАЯ величина, разрешение зазора для них не нужно (в отличие от рабочей точки
#   магнита). Геометрия та же, отличается ТОЛЬКО плотность сетки.
IM8008_COARSE = dataclasses.replace(IM8008, mesh_size=1.6e-3, mesh_size_by_region=None)

STATOR_KINDS = {                    # статор: что было и что стало
    "steel10": lambda: SteinmetzCoefficients.steel10_laminated(LAMINATION),
    "silicon": lambda: SteinmetzCoefficients.silicon_steel_laminated(LAMINATION),
}
ROTOR_COEFFS = SteinmetzCoefficients.steel10_laminated(LAMINATION)   # кольцо под магнитами — массив
ROTOR_SIGMA = 1.0 / 0.14e-6


def P(*a) -> None:
    print(*a, flush=True)


def build(params=IM8008):
    sc = machine_scenario(params, n42sh_magnet((1.0, 0.0, 0.0)),
                          cfg.STEEL_KINDS["steel10"]["curve"]())
    sol = sc.solve(T=20.0, i_peak=0.0, gamma_elec=0.0, turns_per_slot=10.0, max_iter=400)
    turns = ke_from_kv(KV, "bus_sixstep") / (sc.back_emf_constant(sol, turns_per_slot=10.0) / 10.0)
    return sc, float(turns)


# ----------------------------------------------------------------------------- кэш
def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:          # noqa: BLE001 — нет кэша или он битый: просто пересчитать
        return {}


def _save_cache(key: str, value) -> None:
    cache = _load_cache()
    cache[key] = value
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


_VOLUMES: dict[int, np.ndarray] = {}


def _cell_volumes(sc) -> np.ndarray:
    v = _VOLUMES.get(id(sc))
    if v is None:
        geo = sc.geometry
        v = np.array([geo.mesh.cell_area(c) for c in range(geo.mesh.n_cells)], float) * IM8008.axial_length
        _VOLUMES[id(sc)] = v
    return v


def _rotor_wave(sc, turns, *, i_peak, gamma, span, positions) -> np.ndarray:
    """
    Волна B в системе ротора (магнит + кольцо) за пролёт `span` — дорогая часть (нелинейное
    решение на каждом положении). Кэшируется ВОЛНА, а не число потерь: формулы потерь можно
    менять без новых свипов, а устаревшее число из кэша взять невозможно (Л-80).
    """
    mag_idx, ry_idx = rotor_side_cells(sc.geometry)
    idx = np.concatenate([mag_idx, ry_idx])
    loaded = i_peak != 0.0
    key = "wave|cells=%d|span=%.6f|N=%d|i=%s|g=%s|turns=%s|relax=0.2|iter=300|T=20" % (
        sc.geometry.mesh.n_cells, span, positions, repr(float(i_peak)) if loaded else "0",
        repr(float(gamma)) if loaded else "-", repr(float(turns)) if loaded else "-")
    path = WAVES / ("%s.npz" % hashlib.sha1(key.encode("utf-8")).hexdigest()[:16])
    if path.exists():
        d = np.load(path, allow_pickle=False)
        if str(d["key"]) == key and np.array_equal(d["idx"], idx):
            return d["B"]
    t0 = time.time()
    B, _ = rotor_frame_B_series(
        sc.geometry.params, sc.magnet, sc.steel, speed_rpm=1.0, cell_idx=idx, i_peak=i_peak,
        gamma_elec=gamma, turns_per_slot=turns, T=20.0, mech_span=span, n_positions=positions,
        relaxation=0.2, max_iter=300)
    WAVES.mkdir(exist_ok=True)
    np.savez_compressed(path, B=B, idx=idx, key=np.array(key))
    P("   свип ротора (%s): %d положений за %.0f с"
      % ("ток %.1f А" % i_peak if loaded else "холостой ход", positions, time.time() - t0))
    return B


def _rotor_loss(sc, B, n, span) -> float:
    """Потери ротор-стороны [Вт] по волне за пролёт `span` при n об/мин."""
    f_ref = (2.0 * math.pi * n / 60.0) / span
    q = rotor_side_loss_density_from_waveform(
        sc.geometry, B, f_ref, sigma_pm=cfg.magnet_conductivity("ndfeb", 20.0),
        magnet_seg_width=magnet_segment_width(sc.geometry.params, N_SEG), steinmetz=ROTOR_COEFFS,
        rotor_solid=True, sigma_rotor=ROTOR_SIGMA, rotor_mu_r=500.0)
    return float((q * _cell_volumes(sc)).sum())


def _rotor_parts(sc, B, n, span) -> tuple[float, float, float]:
    """Раскладка потерь ротора [Вт]: вихревые магнита, вихревые кольца, гистерезис кольца."""
    mag_idx, ry_idx = rotor_side_cells(sc.geometry)
    nc, nm = sc.geometry.mesh.n_cells, mag_idx.size
    f_ref = (2.0 * math.pi * n / 60.0) / span
    vol = _cell_volumes(sc)
    q_m = magnet_eddy_loss_density_from_waveform(
        B[:, :nm], mag_idx, nc, freq=f_ref, sigma=cfg.magnet_conductivity("ndfeb", 20.0),
        seg_width=magnet_segment_width(sc.geometry.params, N_SEG))
    q_h = hysteresis_density_minor_loops(B[:, nm:], ry_idx, nc, freq=f_ref, coeffs=ROTOR_COEFFS)
    q_r = solid_steel_loss_density_from_waveform(
        B[:, nm:], ry_idx, nc, freq=f_ref, sigma=ROTOR_SIGMA,
        thickness=sc.geometry.params.h_rotor_yoke, mu_r=500.0, coeffs=ROTOR_COEFFS)
    pm, ph, pr = (float((q * vol).sum()) for q in (q_m, q_h, q_r))
    return pm, pr - ph, ph


def _stator_loss(sc, n, turns, kind, *, i_peak, gamma, positions) -> float:
    q, _ = stator_iron_loss_density(
        sc.geometry.params, sc.magnet, sc.steel, speed_rpm=n, i_peak=i_peak, gamma_elec=gamma,
        turns_per_slot=turns, T=20.0, coeffs=STATOR_KINDS[kind](), n_positions=positions,
        relaxation=0.2, max_iter=300)
    return float((q * _cell_volumes(sc)).sum())


# ----------------------------------------------------------------------------- холостой ход
def core_losses(sc, turns: float) -> dict:
    """
    Потери СТАТОРА на холостом ходу при двух скоростях (шаги 0–3), для каждой стали.
    Ротор здесь не считается: его потери на холостом ходу берутся прямо из волны при каждой
    скорости (`_rotor_wave` + `_rotor_loss`) — волна от скорости не зависит.
    """
    key = "stator_noload|cells=%d|turns=%.3f|speeds=%s|lam=%.2e" % (
        sc.geometry.mesh.n_cells, turns, SPEEDS_FIT, LAMINATION)
    cached = _load_cache().get(key)
    if cached:
        P("   статор, холостой ход — из кэша (%s)" % CACHE.name)
        return cached
    params = sc.geometry.params
    out = {"f": []} | {k: [] for k in STATOR_KINDS}
    for n in SPEEDS_FIT:
        f = n / 60.0 * (params.n_poles // 2)
        out["f"].append(f)
        for kind in STATOR_KINDS:
            out[kind].append(_stator_loss(sc, n, turns, kind, i_peak=0.0, gamma=0.0, positions=8))
        P("   свип статора %.0f об/мин (f = %.0f Гц): %s"
          % (n, f, ", ".join("%s %.1f Вт" % (k, out[k][-1]) for k in STATOR_KINDS)))
    _save_cache(key, out)
    return out


def fit_kf(f, p) -> tuple[float, float]:
    """P(f) = k1·f + k2·f² — гистерезис ∝ f, вихревые ∝ f²; две точки, два коэффициента."""
    A = np.vstack([np.asarray(f), np.asarray(f) ** 2]).T
    k1, k2 = np.linalg.lstsq(A, np.asarray(p), rcond=None)[0]
    return float(k1), float(k2)


# ----------------------------------------------------------------------------- шаг 4
def max_torque_gamma(sc, turns: float, i_pk: float) -> tuple[float, float]:
    """
    Угол тока максимального момента: 24 угла по кругу + парабола по трём соседям.
    ⚠ Именно ИЩЕТСЯ: на выдуманной машине подстановка π/2 давала ОТРИЦАТЕЛЬНЫЙ момент.
    """
    key = "gamma|cells=%d|turns=%.3f|i=%.2f" % (sc.geometry.mesh.n_cells, turns, i_pk)
    cached = _load_cache().get(key)
    if cached:
        return float(cached[0]), float(cached[1])
    gs = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    tq = np.array([sc.torque(sc.solve(T=20.0, i_peak=i_pk, gamma_elec=float(g),
                                      turns_per_slot=turns, max_iter=300)) for g in gs])
    k = int(np.argmax(tq))
    ym, y0, yp = tq[(k - 1) % 24], tq[k], tq[(k + 1) % 24]
    g = float((gs[k] + 0.5 * (ym - yp) / (ym - 2.0 * y0 + yp) * (gs[1] - gs[0])) % (2.0 * math.pi))
    t = float(sc.torque(sc.solve(T=20.0, i_peak=i_pk, gamma_elec=g, turns_per_slot=turns,
                                 max_iter=300)))
    _save_cache(key, [g, t])
    return g, t


def loaded_core_losses(sc, turns: float, gamma: float, ops, kt: float) -> dict:
    """
    Потери ядра В РАБОЧИХ ТОЧКАХ винтовой характеристики: ток = M/K_t, угол = γ_max.
    Статор (на электрическом периоде — там поле периодично при любом токе) и момент — кэш
    чисел: их формулы не менялись. Ротор — из кэша ВОЛН на периоде обмотки (90°), потери
    считаются заново при каждом запуске.
    """
    key = "stator_loaded|cells=%d|turns=%.3f|g=%.4f|ops=%s|ns=%d" % (
        sc.geometry.mesh.n_cells, turns, gamma, [round(n) for n, _ in ops], N_STATOR_POS)
    st = _load_cache().get(key)
    params = sc.geometry.params
    if st:
        P("   статор и момент под нагрузкой — из кэша (%s)" % CACHE.name)
    else:
        st = {"n": [], "f": [], "i": [], "M_meas": [], "M_fem": [], "stator": []}
        for n, M in ops:
            i = M / kt
            m_fem = float(sc.torque(sc.solve(T=20.0, i_peak=i, gamma_elec=gamma,
                                             turns_per_slot=turns, max_iter=300)))
            p_st = _stator_loss(sc, n, turns, "silicon", i_peak=i, gamma=gamma, positions=N_STATOR_POS)
            for k_, v in zip(("n", "f", "i", "M_meas", "M_fem", "stator"),
                             (n, n / 60.0 * (params.n_poles // 2), i, M, m_fem, p_st)):
                st[k_].append(float(v))
        _save_cache(key, st)
    out = {k: list(v) for k, v in st.items()}
    out["rotor"], out["parts"] = [], []
    for n, i, f, M, mf, ps in zip(st["n"], st["i"], st["f"], st["M_meas"], st["M_fem"], st["stator"]):
        B = _rotor_wave(sc, turns, i_peak=i, gamma=gamma, span=WINDING_PERIOD, positions=N_ROTOR_POS)
        p_rot = _rotor_loss(sc, B, n, WINDING_PERIOD)
        parts = _rotor_parts(sc, B, n, WINDING_PERIOD)
        assert abs(p_rot - sum(parts)) <= 1e-9 * max(1.0, p_rot)
        out["rotor"].append(p_rot)
        out["parts"].append(parts)
        P("   %4.0f об/мин (f = %4.0f Гц), ток %.1f А: момент расчётный %.3f / измеренный %.3f Н·м;"
          " ротор %.1f Вт (магнит %.1f, кольцо вихр. %.1f, кольцо гист. %.2f), статор %.1f Вт"
          % (n, f, i, mf, M, p_rot, *parts, ps))
    return out


def loginterp(fq: float, f_pts, v_pts) -> float:
    """Интерполяция в двойном логарифмическом масштабе (форму не навязывает; вне — край)."""
    return float(np.exp(np.interp(math.log(fq), np.log(f_pts), np.log(v_pts))))


# ----------------------------------------------------------------------------- намотка
def winding_variants(sc, turns: float) -> list[tuple[str, float, int]]:
    """
    Правдоподобные варианты намотки для оценки эффекта близости — ОБМОТКА НЕ ПУБЛИКУЕТСЯ.
    Двухслойная зубцовая катушка: сторона катушки занимает полпаза, в ней turns/2 проводников.
    Возвращает (название, диаметр жилы, число слоёв поперёк поля рассеяния паза).
    """
    p = sc.geometry.params
    a_c = SLOT_FILL * float(np.mean(slot_areas(sc.geometry))) / turns        # медь на проводник
    slot_ang = (1.0 - p.tooth_width_frac) * 2.0 * math.pi / p.n_slots
    half_w = 0.5 * slot_ang * 0.5 * (p.R_sy + p.R_s_out)                     # ширина полпаза
    side = turns / 2.0
    d1 = math.sqrt(4.0 * a_c / math.pi)
    out = [("один провод Ø%.2f мм" % (d1 * 1e3), d1, max(1, round(side)))]
    for ds in (0.4e-3, 0.3e-3):
        n_str = a_c / (math.pi * ds * ds / 4.0)
        across = max(1, int(half_w / ds))
        out.append(("жгут %.0f×Ø%.1f мм" % (n_str, ds * 1e3), ds,
                    max(1, math.ceil(side * n_str / across))))
    return out


def ac_band(variants, f: float) -> tuple[float, float]:
    fr = [float(dowell_ac_factor(round_wire_delta(strand_diameter=d, freq=f, T=T_CU), m))
          for _, d, m in variants]
    return min(fr), max(fr)


# ----------------------------------------------------------------------------- сверка
def main() -> None:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    P("=" * 110)
    P("СВЕРКА МОДЕЛИ С ИЗМЕРЕНИЯМИ Scorpion IM-8008-100kv — ПРОХОДЫ 2–4 (%d точек отчёта)" % len(rows))
    P("=" * 110)

    sc, turns = build()
    ke = ke_from_kv(KV, "bus_sixstep")
    kt = 1.5 * ke                                           # на АМПЛИТУДУ фазного тока
    l_end = tooth_coil_end_length(sc.geometry)
    L = IM8008.axial_length
    R_act = phase_resistance(sc.geometry, turns_per_slot=turns, slot_fill=SLOT_FILL, T=T_CU)
    R_end = phase_resistance(sc.geometry, turns_per_slot=turns, slot_fill=SLOT_FILL, T=T_CU,
                             end_length_per_turn=l_end)
    variants = winding_variants(sc, turns)
    P("модель: витков %.2f, K_e = %.5f В·с/рад, K_t = %.5f Н·м/А" % (turns, ke, kt))
    P("лобовая часть на виток %.1f мм при активной длине 2×%.1f мм ⇒ R растёт в %.2f раза"
      % (l_end * 1e3, L * 1e3, R_end / R_act))
    P("R фазы при %.0f °C: %.4f Ом (активная длина) → %.4f Ом (с лобовыми)" % (T_CU, R_act, R_end))
    P("варианты намотки для эффекта близости: " + "; ".join(
        "%s, %d слоёв" % (nm, m) for nm, _, m in variants))

    # отбраковка самопротиворечивых точек отчёта — по порогу, заданному ДО сверки
    pts = []
    for r in rows:
        n = float(r["speed_rpm"]); M = float(r["torque_Nm"])
        w = 2.0 * math.pi * n / 60.0
        p_rep = float(r["P_mech_W"])
        pts.append(dict(n=n, M=M, U=float(r["voltage_V"]), I=float(r["current_A"]),
                        eta=float(r["eff_motor_esc_pct"]), p_el=float(r["P_elec_W"]),
                        p_mech=M * w, f=n / 60.0 * (IM8008.n_poles // 2),
                        ok=abs(M * w - p_rep) / p_rep <= CONSISTENCY_TOL,
                        thr=float(r["throttle_us"])))
    good = [p for p in pts if p["ok"]]
    bad = [p for p in pts if not p["ok"]]
    P("самопротиворечивые точки отчёта (|M·ω − P_мех| > %.0f %%): %s"
      % (100 * CONSISTENCY_TOL, ", ".join("%.0f об/мин" % p["n"] for p in bad) or "нет"))

    # ================================ ЭТАП А: баланс мощности без потерь в ядре
    P("\n" + "=" * 110)
    P("ЭТАП А — ТОК ШИНЫ ИЗ БАЛАНСА МОЩНОСТИ: I = (P_мех + P_медь)/U. Потерь в ядре здесь НЕТ,")
    P("поэтому модель ОБЯЗАНА недооценивать ток. Глубина модуляции m = (K_e·ω + I·R)/((2/π)·U).")
    P("=" * 110)
    P("%-6s %6s %7s %7s %6s %8s %8s  %s" % ("газ", "об/мин", "M,Н·м", "I изм", "m", "I мод", "разн,%", ""))
    for p in pts:
        iph = p["M"] / kt
        m = (ke * 2 * math.pi * p["n"] / 60 + iph * R_end) / ((2 / math.pi) * p["U"])
        i_mod = (p["p_mech"] + 3.0 * (iph / math.sqrt(2)) ** 2 * R_end) / p["U"]
        p["dA"] = 100.0 * (i_mod - p["I"]) / p["I"]
        P("%-6.0f %6.0f %7.3f %7.2f %6.3f %8.2f %+8.1f  %s"
          % (p["thr"], p["n"], p["M"], p["I"], m, i_mod, p["dA"], "" if p["ok"] else "⚠ исключена"))
    dA = np.array([p["dA"] for p in good])
    P("невязка по току (годные точки): среднее %+.1f %%, размах %+.1f…%+.1f %% — "
      "ожидается отрицательной всюду" % (dA.mean(), dA.min(), dA.max()))

    # ================================ ЭТАП Б: потери и КПД — абляция
    P("\n" + "=" * 110)
    P("ЭТАП Б — КПД, АБЛЯЦИЯ ПО ШАГАМ")
    P("=" * 110)
    sc_c, turns_c = build(IM8008_COARSE)
    P("сетка для свипов потерь: %d ячеек (боевая %d); витки на грубой сетке %.2f против %.2f"
      % (sc_c.geometry.mesh.n_cells, sc.geometry.mesh.n_cells, turns_c, turns))
    cl = core_losses(sc_c, turns_c)
    ks = {k: fit_kf(cl["f"], cl[k]) for k in STATOR_KINDS}
    for k, (a, b) in ks.items():
        P("   статор %-8s, холостой ход      : P = %.4g·f + %.4g·f²" % (k, a, b))
    # ротор на холостом ходу — прямо из волны: она от скорости не зависит, меняется только частота
    B_nl = _rotor_wave(sc_c, turns_c, i_peak=0.0, gamma=0.0, span=WINDING_PERIOD, positions=N_ROTOR_POS)
    rot_nl: dict[float, float] = {}

    def rotor_noload(f: float) -> float:
        if f not in rot_nl:
            rot_nl[f] = _rotor_loss(sc_c, B_nl, 60.0 * f / (IM8008.n_poles // 2), WINDING_PERIOD)
        return rot_nl[f]

    P("   ротор, холостой ход — прямо из волны (пролёт %.0f°, %d положений): %s"
      % (math.degrees(WINDING_PERIOD), N_ROTOR_POS,
         ", ".join("%.0f Гц → %.2f Вт" % (f, rotor_noload(f)) for f in cl["f"])))

    # ---- шаг 4: потери под нагрузкой
    P("\nШАГ 4 — ПОТЕРИ ЯДРА ПОД НАГРУЗКОЙ (реакция якоря), свип ротора на %.0f°, %d положений"
      % (math.degrees(WINDING_PERIOD), N_ROTOR_POS))
    P("ПРЕДСКАЗАНИЕ, зафиксированное до прогона: КПД регулятора %.2f…%.2f во всех годных точках,"
      " |тренд| < %.1f п.п./кГц" % (PREDICTION_ESC[0], PREDICTION_ESC[1], PREDICTION_SLOPE))
    ops = []
    for target in OP_RPMS:
        q = min(good, key=lambda z: abs(z["n"] - target))
        ops.append((q["n"], q["M"]))
    gamma, m_g = max_torque_gamma(sc_c, turns_c, ops[-1][1] / kt)
    P("   угол тока максимального момента γ = %.1f° (момент %.3f Н·м при токе %.1f А)"
      % (math.degrees(gamma), m_g, ops[-1][1] / kt))
    ld = loaded_core_losses(sc_c, turns_c, gamma, ops, kt)
    # КОНТРОЛЬ МЕТОДИКИ: на холостом ходу волна периодична по зубцовому делению ⇒ пролёт 90° и
    # каждое из девяти зубцовых окон той же волны обязаны дать одно и то же с точностью до шума
    # пересетки. Критерий задан ДО прогона: значение на 90° — внутри разброса девяти окон.
    n_top = ld["n"][-1]
    k_win = int(round(N_ROTOR_POS * SLOT_PITCH / WINDING_PERIOD))
    p90 = _rotor_loss(sc_c, B_nl, n_top, WINDING_PERIOD)
    wins = [_rotor_loss(sc_c, B_nl[j * k_win:(j + 1) * k_win], n_top, SLOT_PITCH)
            for j in range(N_ROTOR_POS // k_win)]
    parts = _rotor_parts(sc_c, B_nl, n_top, WINDING_PERIOD)
    P("   КОНТРОЛЬ МЕТОДИКИ (холостой ход, %.0f об/мин): пролёт 90° — %.2f Вт (магнит %.2f, кольцо"
      " вихр. %.2f, кольцо гист. %.3f); девять окон 10° — %.2f…%.2f Вт, среднее %.2f ⇒ %s"
      % (n_top, p90, *parts, min(wins), max(wins), float(np.mean(wins)),
         "СОШЛОСЬ" if min(wins) <= p90 <= max(wins) else "НЕ СОШЛОСЬ"))
    P("   K_t ПОД НАГРУЗКОЙ: расчётный момент при токе M/K_t против измеренного — "
      + ", ".join("%.0f: %+.1f %%" % (n, 100 * (mf / mm - 1))
                  for n, mf, mm in zip(ld["n"], ld["M_fem"], ld["M_meas"])))

    def core_at(source: str, f: float) -> float:
        if source == "loaded":
            return loginterp(f, ld["f"], ld["rotor"]) + loginterp(f, ld["f"], ld["stator"])
        a1, a2 = ks[source]
        return rotor_noload(f) + (a1 * f + a2 * f * f)

    def model_eta(p, source, R, ac, bound):
        f = p["f"]
        p_core = core_at(source, f)
        iph = p["M"] / kt
        if ac:
            F = ac_band(variants, f)[bound]
            R = phase_resistance(sc.geometry, turns_per_slot=turns, slot_fill=SLOT_FILL,
                                 T=T_CU, end_length_per_turn=l_end, ac_factor=F)
        p_cu = 3.0 * (iph / math.sqrt(2)) ** 2 * R
        return 100.0 * p["p_mech"] / (p["p_mech"] + p_cu + p_core), p_cu, p_core

    steps = [
        ("0 как в проходе 1", "steel10", R_act, None),
        ("1 + кремнистая сталь", "silicon", R_act, None),
        ("2 + лобовые части", "silicon", R_end, None),
        ("3 + эффект близости", "silicon", None, "ac"),
        ("4 + потери под нагрузкой", "loaded", None, "ac"),
    ]
    fk = np.array([p["f"] for p in good]) / 1000.0
    P("\n%-30s %9s %15s %11s %17s %s" % ("шаг", "среднее", "размах, п.п.", "тренд",
                                         "КПД регулятора", "ядро ≤ всех потерь при 439 Гц"))
    P("-" * 116)
    verdict = []
    for name, source, R, ac in steps:
        for bound in ((0, 1) if ac else (0,)):
            d, esc = [], []
            for p in good:
                e, _, _ = model_eta(p, source, R, ac, bound)
                d.append(e - p["eta"]); esc.append(p["eta"] / e)
            d, esc = np.array(d), np.array(esc)
            slope = np.polyfit(fk, d, 1)[0]
            p0 = good[0]
            core0 = core_at(source, p0["f"])
            P("%-30s %+8.1f  %+6.1f…%+6.1f %+8.1f/кГц  %6.3f…%6.3f %s  %s"
              % (name + ("" if not ac else (" (мин)" if bound == 0 else " (макс)")),
                 d.mean(), d.min(), d.max(), slope, esc.min(), esc.max(),
                 "⚠ >1" if esc.max() > 1.0 else "    ",
                 "да" if core0 <= p0["p_el"] - p0["p_mech"] else "НЕТ — невозможно"))
            if source == "loaded":
                verdict.append((bound, esc.min(), esc.max(), slope))

    # ---- поточечная раскладка потерь шага 4 (нижняя граница эффекта близости)
    P("\nРАСКЛАДКА ПОТЕРЬ, шаг 4 (эффект близости — нижняя граница), Вт:")
    P("%-6s %6s %7s | %7s %8s %8s %8s | %7s %7s %7s | %s"
      % ("об/мин", "f,Гц", "все изм", "медь", "ядро ст", "ротор", "остаток", "КПД изм",
         "КПД мод", "регул.", ""))
    rest, fr_ = [], []
    for p in pts:
        f = p["f"]
        lo = model_eta(p, "loaded", None, "ac", 0)
        stat = loginterp(f, ld["f"], ld["stator"])
        rot = loginterp(f, ld["f"], ld["rotor"])
        tot = p["p_el"] - p["p_mech"]
        remain = tot - lo[1] - stat - rot
        if p["ok"]:
            rest.append(remain); fr_.append(f)
        P("%-6.0f %6.0f %7.1f | %7.1f %8.1f %8.1f %8.1f | %7.1f %7.1f %7.3f | %s"
          % (p["n"], f, tot, lo[1], stat, rot, remain, p["eta"], lo[0],
             p["eta"] / lo[0], "" if p["ok"] else "⚠ исключена (и вне диапазона интерполяции)"))
    rest, fr_ = np.array(rest), np.array(fr_)
    pos = rest > 0
    if pos.sum() >= 3:
        P("\nОСТАТОК после шага 4 растёт по частоте как f^%.2f (по %d точкам, где он > 0), "
          "максимум %.1f Вт." % (np.polyfit(np.log(fr_[pos]), np.log(rest[pos]), 1)[0],
                                  pos.sum(), rest.max()))

    # ---- проверка предсказания
    P("\n" + "=" * 110)
    P("ПРОВЕРКА ПРЕДСКАЗАНИЯ (критерии заданы ДО прогона):")
    for bound, e_lo, e_hi, slope in verdict:
        ok_esc = PREDICTION_ESC[0] <= e_lo and e_hi <= PREDICTION_ESC[1]
        ok_slope = abs(slope) < PREDICTION_SLOPE
        P("   эффект близости %s: КПД регулятора %.3f…%.3f [%s], тренд %+.1f п.п./кГц [%s] ⇒ %s"
          % ("мин" if bound == 0 else "макс", e_lo, e_hi, "в полосе" if ok_esc else "ВНЕ полосы",
             slope, "ок" if ok_slope else "НЕ ок",
             "ПОДТВЕРЖДЕНО" if (ok_esc and ok_slope) else "ОПРОВЕРГНУТО"))
    P("=" * 110)


def precompute_wave(k: int) -> None:
    """
    Досчитать ОДНУ волну ротора — для параллельного запуска несколькими процессами (свип на
    72 положения идёт около часа). k = −1 — холостой ход; k = 0…3 — рабочие точки с теми же
    током и углом, что возьмёт main() (из кэша статора и угла тока прохода 3).
    """
    sc_c, turns_c = build(IM8008_COARSE)
    if k < 0:
        _rotor_wave(sc_c, turns_c, i_peak=0.0, gamma=0.0, span=WINDING_PERIOD, positions=N_ROTOR_POS)
        return
    cache = _load_cache()
    head = "cells=%d|turns=%.3f|" % (sc_c.geometry.mesh.n_cells, turns_c)
    gamma = next(float(v[0]) for kk, v in cache.items() if kk.startswith("gamma|" + head))
    st = next(v for kk, v in cache.items()
              if kk.startswith("stator_loaded|" + head + "g=%.4f|" % gamma))
    _rotor_wave(sc_c, turns_c, i_peak=st["i"][k], gamma=gamma, span=WINDING_PERIOD,
                positions=N_ROTOR_POS)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--wave":
        precompute_wave(int(sys.argv[2]))
    else:
        main()
