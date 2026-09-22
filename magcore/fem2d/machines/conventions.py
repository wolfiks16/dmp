# -*- coding: utf-8 -*-
"""
ЭЛЕКТРИЧЕСКИЕ КОНВЕНЦИИ: паспортное kV ↔ модельные K_e/K_t, потолок инвертора.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Аудит 2026-09-03 нашёл, что в сценарии стояло
`K_e = 60/(2π·kV)`. Это НЕ амплитуда фазной ЭДС трёхфазной машины: формула строга для
КОЛЛЕКТОРНОЙ машины постоянного тока, а для бесколлекторной она даёт момент на ампер тока
ЗВЕНА (вывод Drela, MIT 2007, из энергобаланса K_Q = K_V). Сравнение модельного
K_e = p·λ_m с этой величиной занижало расхождение в 1,7 раза и увело диагноз в сторону.

ГЛАВНОЕ ПРАКТИЧЕСКОЕ ПРАВИЛО. Паспортное kV — величина с конвенционной неопределённостью
(разные производители меряют по шине / по линейному / действующее / амплитудное; разброс
до 1,4 раза, а «звезда против треугольника» — ещё √3). Поэтому **проектную проверку строить
на kV НЕЛЬЗЯ**. Физически однозначное ограничение — ПОТОЛОК НАПРЯЖЕНИЯ:

    K_e^{ф,пик} · ω_мех + (падения) ≤ V̂_ф,max = k_мод · U_шины.

Перевод kV нужен только для сверки с паспортом и для отчётов, и всегда с ЯВНО указанной
конвенцией.

ВЫВОД МНОЖИТЕЛЕЙ. Пусть λ_m — амплитуда потокосцепления ПМ на фазу (Кларк с сохранением
АМПЛИТУДЫ), p — число пар полюсов, ω_м — механическая скорость. Тогда
    K_e^{ф,пик} = p·λ_m              (амплитуда ФАЗНОЙ ЭДС на рад/с),
    K_t         = (3/2)·p·λ_m        (момент на АМПЛИТУДУ фазного тока; 3/2 = три фазы × ½),
    Ê_лин       = √3·Ê_ф             (разность двух синусоид со сдвигом 120°).
Определение паспорта: n [об/мин] = kV · V, то есть V = ω_м·(60/2π)/kV. Подставляя, какое
именно V имел в виду производитель, получаем множитель C: K_e^{ф,пик} = C·(60/2π)/kV.

⚠ «K_e = K_t» справедливо ТОЛЬКО в преобразовании Парка с сохранением МОЩНОСТИ (обе равны
√(3/2)·p·λ_m). В нашей амплитудной системе они отличаются в 3/2 раза — именно эта ловушка
и стояла в комментарии сценария.
"""
from __future__ import annotations

import math

# --- kV → K_e^{ф,пик}: множитель C в K_e = C·(60/2π)/kV -----------------------------------
# Название → (множитель, пояснение, K_e·kV для сверки)
KV_CONVENTIONS: dict[str, tuple[float, str]] = {
    # Паспорт «от батареи» с трапецеидальным регулятором: первая гармоника фазного
    # напряжения при шеститактной коммутации = (2/π)·U_шины.  ДЕФОЛТ: так пишет
    # большинство паспортов БК-двигателей для БПЛА.
    "bus_sixstep": (2.0 / math.pi, "шина U_dc, шеститактный регулятор"),
    # Векторное управление: предел линейной зоны V̂_ф = U_dc/√3, при этом V̂_лин = U_dc ровно,
    # поэтому конвенция ТОЖДЕСТВЕННА «линейное пиковое».
    "bus_svpwm": (1.0 / math.sqrt(3.0), "шина U_dc, векторная ШИМ"),
    "ll_peak": (1.0 / math.sqrt(3.0), "линейное ПИКОВОЕ (так задаёт Motor Control Blockset)"),
    "ll_rms": (math.sqrt(2.0) / math.sqrt(3.0), "линейное ДЕЙСТВУЮЩЕЕ (мультиметр между фазами)"),
    "phase_peak": (1.0, "фазное ПИКОВОЕ"),
    "phase_rms": (math.sqrt(2.0), "фазное ДЕЙСТВУЮЩЕЕ"),
}
KV_CONVENTION_DEFAULT = "bus_sixstep"

# --- потолок инвертора: амплитуда ПЕРВОЙ ГАРМОНИКИ фазного напряжения от шины U_dc --------
MODULATION_LIMITS: dict[str, tuple[float, str]] = {
    "spwm": (0.5, "синусоидальная ШИМ, линейная зона"),
    "svpwm": (1.0 / math.sqrt(3.0), "векторная ШИМ / инжекция 3-й гармоники (+15,5 %)"),
    "six_step": (2.0 / math.pi, "шеститактная коммутация — абсолютный максимум"),
}
MODULATION_DEFAULT = "svpwm"

_RPM_TO_RAD = 60.0 / (2.0 * math.pi)     # = 9.5493


def _factor(convention: str) -> float:
    try:
        return KV_CONVENTIONS[convention][0]
    except KeyError:
        raise ValueError("неизвестная конвенция kV %r; допустимо: %s"
                         % (convention, ", ".join(sorted(KV_CONVENTIONS)))) from None


def ke_from_kv(kv_rpm_per_volt: float, convention: str = KV_CONVENTION_DEFAULT) -> float:
    """Паспортное kV [об/(мин·В)] → K_e = амплитуда ФАЗНОЙ ЭДС на рад/с механические."""
    if not (kv_rpm_per_volt > 0.0):
        raise ValueError("kV must be positive.")
    return _factor(convention) * _RPM_TO_RAD / float(kv_rpm_per_volt)


def kv_from_ke(ke_phase_peak: float, convention: str = KV_CONVENTION_DEFAULT) -> float:
    """Обратный перевод — для отчётов («эквивалентное kV этой обмотки»)."""
    if not (ke_phase_peak > 0.0):
        raise ValueError("ke must be positive.")
    return _factor(convention) * _RPM_TO_RAD / float(ke_phase_peak)


def kv_spread(kv_rpm_per_volt: float) -> dict[str, float]:
    """K_e по ВСЕМ конвенциям — честная мера неопределённости паспортного числа."""
    return {name: ke_from_kv(kv_rpm_per_volt, name) for name in KV_CONVENTIONS}


def phase_voltage_limit(u_dc: float, modulation: str = MODULATION_DEFAULT) -> float:
    """Максимальная амплитуда первой гармоники ФАЗНОГО напряжения от шины U_dc [В]."""
    try:
        k = MODULATION_LIMITS[modulation][0]
    except KeyError:
        raise ValueError("неизвестная модуляция %r; допустимо: %s"
                         % (modulation, ", ".join(sorted(MODULATION_LIMITS)))) from None
    if not (u_dc > 0.0):
        raise ValueError("u_dc must be positive.")
    return k * float(u_dc)


def max_speed_rpm(ke_phase_peak: float, u_dc: float,
                  modulation: str = MODULATION_DEFAULT) -> float:
    """
    Предельная скорость ХОЛОСТОГО ХОДА [об/мин] для данной обмотки и шины.

    Из K_e·ω_м ≤ V̂_ф,max. Под нагрузкой ещё вычитаются I·R и реактивное падение ω_э·L_s·I,
    поэтому это ОЦЕНКА СВЕРХУ: реальная рабочая скорость ниже.
    """
    if not (ke_phase_peak > 0.0):
        raise ValueError("ke must be positive.")
    return phase_voltage_limit(u_dc, modulation) / ke_phase_peak * _RPM_TO_RAD


def turns_for_speed(ke_at_turns: float, turns: float, *, speed_rpm: float, u_dc: float,
                    modulation: str = MODULATION_DEFAULT, margin: float = 1.0) -> float:
    """
    Сколько витков на паз допускает шина при требуемой скорости (K_e линейна по виткам).

    `margin` > 1 — запас на падения (I·R, ω_э·L_s·I, просадка батареи); при margin=1,25
    остаётся 20 % напряжения на всё это.
    """
    if not (turns > 0.0 and ke_at_turns > 0.0 and speed_rpm > 0.0 and margin >= 1.0):
        raise ValueError("turns, ke, speed must be positive; margin >= 1.")
    omega = float(speed_rpm) / _RPM_TO_RAD
    ke_max = phase_voltage_limit(u_dc, modulation) / (omega * float(margin))
    return float(turns) * ke_max / float(ke_at_turns)
