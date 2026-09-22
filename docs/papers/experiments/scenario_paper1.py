"""
P-B0 — ПРЕД-РЕГИСТРАЦИЯ сценария Статьи 1 (ядро К6', B-full).

Единственное место с ЗАФИКСИРОВАННЫМИ условиями расчёта (методология §12): все параметры
заданы ДО прогонов, каждый с источником. Материальные константы — deep-research
(`docs/math/bfull_parameters.md`); мотор — спецификация Sergey (2026-08-03); проектные
дефолты помечены [DEFAULT] (заменить под реальный мотор при уточнении).

Оракул P-B0 (запуск модуля): validate() проходит + печать пред-регистрированной сводки
⇒ условия зафиксированы до знания ответа. Физику (потери/ГУ/зазор) подключают P-B1..B7,
читая ЭТИ значения; сам конфиг ответа не предрешает.

Запуск:  PYTHONPATH=<repo> python docs/papers/experiments/scenario_paper1.py
"""
from __future__ import annotations

import dataclasses
import math
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.domain.steel_curves import m270_35a_cogent_bh_curve, steel10_bh_curve
from magcore.fem2d.airgap import airgap_k_eff as _airgap_k_eff  # P-B2
from magcore.fem2d.machines.iron_loss import (
    STEEL10_CP,
    STEEL10_DENSITY,
    STEEL10_K_TH,
    STEEL10_RESISTIVITY,
    SteinmetzCoefficients,
)
from magcore.fem2d.machines.conventions import (  # P-B0': конвенции kV/модуляции
    KV_CONVENTIONS,
    MODULATION_LIMITS,
    ke_from_kv,
    kv_from_ke,
    max_speed_rpm,
)
from magcore.fem2d.machines.excitation import slot_areas
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, Region
from magcore.fem2d.machines.scenario import machine_scenario
from magcore.fem2d.machines.thermal_scenario import MachineThermalProperties
from magcore.fem2d.verification import (
    VerificationReport,
    check_airgap_resolution,
    check_current_density,
    check_permeance_coefficient,
    check_tangential_stress,
    check_thermal_loading,
    check_voltage_headroom,
)

# ============================================================ 1. МОТОР (спец. Sergey 2026-08-03)
KV = 350.0                    # об/мин на вольт [источник: Sergey]
N_SERIES = 6                  # 6S LiPo
U_CELL_NOM = 3.7              # В/ячейку (номинал)
U_DC_NOM = N_SERIES * U_CELL_NOM       # = 22.2 В (номинал)
U_DC_FULL = N_SERIES * 4.2             # = 25.2 В (полный заряд)
P_MAX = 2000.0               # Вт — пиковая мощность [Sergey]

# --- КОНВЕНЦИЯ kV (правка аудита 2026-09-03) --------------------------------------------
# Прежде здесь стояло KE = 60/(2π·KV) с комментарием «= Н·м/А = 1/kV(СИ)». Это формула
# КОЛЛЕКТОРНОЙ машины; для трёхфазной она не соответствует ни одной паспортной конвенции и
# завышала опорное K_e в 1,57 раза (π/2). Правильный перевод — `machines/conventions.py`.
# Выбрана конвенция «от шины, шеститактный регулятор» — так пишет большинство паспортов
# БК-моторов для БПЛА. ⚠ Разброс между конвенциями до 2,4 раза, поэтому ПРОЕКТНАЯ проверка
# ведётся не по kV, а по ПОТОЛКУ НАПРЯЖЕНИЯ (см. validate_design).
KV_CONVENTION = "bus_sixstep"
MODULATION = "six_step"                # тип регулятора; согласован с конвенцией kV
KE_NAMEPLATE = ke_from_kv(KV, KV_CONVENTION)      # ≈0.01737 В·с/рад (амплитуда ФАЗНОЙ ЭДС)

N_NOLOAD = KV * U_DC_NOM               # ≈7770 об/мин ХХ (сходится с «ХХ≥7000»)
N_OPER = 7000.0               # об/мин рабочая (НАГРУЖЕННАЯ ≈0.90·ХХ) [Sergey: ХХ≥7000]
N_POLES = 14
P_POLE_PAIRS = N_POLES // 2            # =7
F_ELEC = N_OPER / 60.0 * P_POLE_PAIRS  # ≈816.7 Гц

# ============================================================ 2. ГЕОМЕТРИЯ (дефолт 12/14, согласовано)
GEOM = OutrunnerPMSMParams(n_slots=12, n_poles=N_POLES, mesh_size=0.0030)
SLOT_FILL = 0.45
GAMMA_ELEC = 0.0              # worst-case d-ось (чисто размагничивающий) [решение Sergey, режим A]

# --- ОБМОТКА: витки заданы ПОТОЛКОМ ШИНЫ, а не «на глаз» (правка аудита) ------------------
# Измерено на сходившейся сетке (магнит+зазор 0,5 мм): K_e = 0,002493 В·с/рад НА ВИТОК.
# Отсюда:
#   · паспортное kV=350 в конвенции «от шины/шеститакт» ⇒ 6,97 витка;
#   · потолок шины 6S при 7000 об/мин ⇒ ≤7,73 (шеститакт), ≤7,01 (SVPWM), ≤6,07 (СШИМ).
# Принято N = 7: ХХ получается 7733 об/мин (паспорт 7770, расхождение 0,5 %), рабочие
# 7000 об/мин = 90 % ХХ — остаётся запас на I·R и ω_э·L_s·I.
# ⚠ ПРЕЖДЕ СТОЯЛО 20 витков. Это давало эквивалентное kV = 122 вместо 350, фазную ЭДС 36 В
#   при потолке шины 14,1 В, то есть машина не раскручивалась выше ~2750 об/мин, и частота
#   817 Гц, на которой посчитаны ВСЕ потери, была недостижима.
# ⚠ ЧУВСТВИТЕЛЬНОСТЬ: при векторном управлении без перемодуляции потолок 7015 об/мин, и
#   7000 под нагрузкой уже недостижимы. Это записано в SENSITIVITY.
KE_PER_TURN_MEASURED = 0.002493        # В·с/рад на виток, сетка critical_mesh(), 20 °C
TURNS_PER_SLOT = 7.0

# ============================================================ 3. СТАЛЬ M270-35A [Cogent datasheet]
STEEL_STEINMETZ = dict(       # P=k_h·f·B^a + k_c·f²·B² + k_ex·(f·B)^1.5 [Вт/кг], фит к Cogent @≤1000Гц
    k_h=1.79e-2, alpha=1.74, k_c=5.04e-5, k_ex=2.5e-4,
)
STEEL_RESISTIVITY = 0.52e-6   # Ом·м [Cogent]
STEEL_DENSITY = 7690.0        # кг/м³ [SIJ]
STEEL_K_TH = 25.0             # Вт/(м·К), в плоскости = 2D радиальный поток [SIJ]
STEEL_CP = 450.0              # Дж/(кг·К) [лит., подстановка]
STEEL_C_VOL = STEEL_DENSITY * STEEL_CP   # ≈3.46e6 Дж/(м³·К)

# ============================================================ 4. МАГНИТЫ [Ruoho / Eclipse / Neorem]
# σ(T) для вихревых потерь. NdFeB: ρ⊥(T)=1.25+0.92e-3·T [µΩм]; SmCo: ~0.85 µΩм (слабая T-завис.).
def magnet_conductivity(material: str, T: float) -> float:
    """Электропроводность магнита σ(T) [См/м]. material: 'ndfeb'|'smco'."""
    if material == "ndfeb":
        rho_uohm = 1.25 + 0.92e-3 * float(T)          # µΩ·м [Ruoho]
    elif material == "smco":
        rho_uohm = 0.85 * (1.0 + 0.05e-2 * (float(T) - 20.0))  # ~0.85 µΩм, α~+0.05%/K [подстановка, LOW]
    else:
        raise ValueError("material must be 'ndfeb' or 'smco'.")
    return 1.0 / (rho_uohm * 1e-6)

MAGNET_PROPS = {   # k[Вт/мК], c_p[Дж/кгК], ρ[кг/м³]  [Neorem/Eclipse]
    "ndfeb": dict(k_th=8.5, cp=450.0, density=7600.0),
    "smco":  dict(k_th=11.6, cp=355.0, density=8400.0),
}
N_SEG = 1                     # [DEFAULT] сегментация магнита (1=сплошной, консервативно) — уточнить у Sergey

# ============================================================ 5. ВОЗДУХ + ТЕЙЛОР-ЗАЗОР [Howey 2010]
K_AIR = 0.030                 # Вт/(м·К) @~90° [Incropera]
NU_AIR = 2.1e-5               # м²/с @~90° [Incropera]
TA_CRIT = 41.19               # критич. число Тейлора [Howey]
F_G = 1.04                    # геом. фактор (узкий зазор) [Howey]

def airgap_k_eff(n_rpm: float) -> float:
    """k_eff(n) зазора по Тейлору–Куэтту при геометрии/воздухе сценария (делегирует
    magcore.fem2d.airgap, P-B2). Заменяет ad-hoc k_gap=0.5 (был завышен ~8-18×)."""
    r_m = 0.5 * (GEOM.R_s_out + GEOM.R_mag_in)         # средний радиус зазора
    return _airgap_k_eff(n_rpm, r_m, GEOM.air_gap, k_air=K_AIR, nu_air=NU_AIR, F_g=F_G)

# ============================================================ 6. ГРАНИЧНЫЕ УСЛОВИЯ [deep-research/§5]
H_OUT = 20.0                  # Вт/(м²·К) обдув ротора, worst-case зависание/жарко [DEFAULT]
H_OUT_FLIGHT = 100.0          # активный полёт (для справки)
H_IN = 800.0                  # статор→рама, сильный контакт (кондукция) [DEFAULT]
T_FRAME = 60.0               # °C рама/крепление [DEFAULT] — уточнить
T_AMB_HOT = 60.0             # °C горячая среда/замачивание (worst-case, режим A) [DEFAULT]

# ============================================================ 7. ЧИСЛЕННОСТЬ [калибровка 2026-08]
# --- ПОЛИТИКА СЕТКИ: магнит и зазор всегда мельчайшие -------------------------------------
# Уровень выбран ПО СХОДИМОСТИ (исследование 2026-08-27, fig_risk_map.py, точка 130 °C / 40 А):
#   равномерн. 4 мм (172 яч.) → падение ЭДС-пост. 6.29 %, повр. объём 54 %
#   магнит+зазор 1 мм  (874) → 4.55 %, 35 %
#   магнит+зазор 0.5 мм (3041) → 3.24 %, 27 %          <- принято
#   зазор 0.25 мм       (5073) → 3.25 %, 27 %  (зазор УЖЕ сошёлся при 0.5 мм)
#   магнит 0.25 мм     (10842) → 3.74 %, 29 %  (+0.5 п.п. при 6-кратной цене — не окупается)
# ⚠ ret_min НЕ сходится (0.741 / 0.600 / 0.592) — это экстремум по ячейкам у кромочной
# особенности; в отчёты идут ИНТЕГРАЛЬНЫЕ величины (момент, ЭДС-пост., повреждённый объём).
MESH_MAGNET = 0.5e-3          # м — размер элемента в магните
# ⚠ ЗАЗОР: размер выводится ИЗ ШИРИНЫ ЗАЗОРА, а не задаётся абсолютным числом.
#   Правка 2026-09-08: стояло 0,5 мм на зазор 1 мм = 2,3 элемента поперёк, то есть
#   расчёт НЕ ПРОХОДИЛ собственную проверку `check_airgap_resolution` (нужно ≥3).
#   Последствие было не теоретическим: коэффициент проницания на нерешённом зазоре
#   занижался — на сетке 3 мм давал P_c = 2,14, на 0,5 мм — 2,69, сошёлся на 2,73.
#   Именно число 2,14 попало в аудит как «цепь склонна к размагничиванию»; настоящее
#   значение ближе к нижней границе нормы, а не вдвое ниже неё.
#   Сходимость (зазор 1 мм): 0,5→2,686 | 0,333→2,701 | 0,25→2,717 | 0,167→2,725 | 0,125→2,732.
MESH_AIR_GAP_PER_GAP = 3.0    # элементов поперёк зазора (минимум по verification.py)


def critical_mesh(geom: OutrunnerPMSMParams | None = None) -> dict:
    """Помельчение по регионам для боевых расчётов: магнит и зазор — мельчайшая сетка."""
    g = GEOM if geom is None else geom
    return {"magnet": MESH_MAGNET, "air_gap": g.air_gap / MESH_AIR_GAP_PER_GAP}


DT = 0.5                      # с — dt-оракул проходит при ≤0.5 (был 51° при dt=5)
T_CAP = 200.0                 # °C — предел изоляции (класс H+запас)
N_STEPS = 240                 # горизонт 120 с при dt=0.5
STEADY_TOL = 0.05             # °C/с — стоп по установившемуся режиму
SUBSTEPS_HEAT = 32
SUBSTEPS_SWEEP = 8

# --- РЕЖИМЫ НАГРУЗКИ (правка аудита 2026-09-03) ------------------------------------------
# ПРЕЖДЕ: SWEEP_CURRENTS = [40…150 А] при 20 витках, что давало 33…125 А/мм² — свип
# начинался НА ПИКЕ и уходил далеко за предел разрушения, а точка 40 А была подписана как
# «номинал длительный». Держать такое 120–700 с не может ни один реальный двигатель, и
# «перегрев» в прежних расчётах объяснялся именно этим.
#
# ТЕПЕРЬ режимы заданы через ПЛОТНОСТЬ ТОКА (величина, для которой есть нормы), а токи из
# неё выводятся: J = N·I_скз/A_меди, A_меди = k_зап·A_паз.
# Нормы: Pyrhönen, «Design of Rotating Electrical Machines», 2-е изд. 2014 —
#   табл. 6.3 (с. 298): машины с ПМ J = 4…6,5 А/мм², для ЗУБЦОВЫХ КАТУШЕК книга
#     предписывает строку полюсной обмотки 2…5,5;
#   ур. (7.17) и пример 7.3 (с. 338): тепловая нагрузка AJ ≤ 42,25·10¹⁰ А²/м³ — ГЛАВНЫЙ
#     книжный критерий длительного режима, от размера машины не зависит.
# ⚠ Полосы ниже ВЫШЕ книжных, и это осознанно: таблицы составлены для ЗАКРЫТЫХ промышленных
#   машин в режиме S1, а у нас открытый обдуваемый аутраннер с изоляцией класса H. Практика
#   этого класса — 12…17 А/мм² длительно. Расхождение НЕ усредняется: в отчёт идут ОБЕ
#   границы, а обоснование выбора — по критерию AJ (он книжный).
# ⚠ «Длительный» у производителей БПЛА-моторов нередко означает 180 с, а не S1.
LOAD_REGIMES = {                       # имя → (J [А/мм², СКЗ], допустимая длительность)
    "continuous": (12.0, None),        # без ограничения
    "overload": (22.0, 180.0),         # ~3 минуты
    "peak": (33.0, 20.0),              # взлёт, 15…30 с
}
REGIME_DEFAULT = "continuous"


_SLOT_COPPER_AREA: float | None = None      # кэш: сетка строится один раз по требованию


def slot_copper_area() -> float:
    """Площадь МЕДИ в пазу [м²] = k_зап · A_паз (сетка строится лениво и кэшируется)."""
    global _SLOT_COPPER_AREA
    if _SLOT_COPPER_AREA is None:
        g = machine_scenario(GEOM, n42sh_magnet((1.0, 0.0, 0.0)),
                             STEEL_KINDS[STEEL_KIND_DEFAULT]["curve"]()).geometry
        _SLOT_COPPER_AREA = SLOT_FILL * float(np.mean(slot_areas(g)))
    return _SLOT_COPPER_AREA


def current_for_density(j_a_per_mm2: float, turns: float | None = None) -> float:
    """Амплитуда фазного тока [А], дающая заданную плотность тока в меди [А/мм², СКЗ]."""
    n = TURNS_PER_SLOT if turns is None else float(turns)
    return float(j_a_per_mm2) * 1e6 * slot_copper_area() / n * math.sqrt(2.0)


def current_density(i_peak: float, turns: float | None = None) -> float:
    """Обратное: плотность тока в меди [А/мм², СКЗ] при заданной амплитуде фазного тока."""
    n = TURNS_PER_SLOT if turns is None else float(turns)
    return n * (float(i_peak) / math.sqrt(2.0)) / slot_copper_area() / 1e6


def linear_current_density(i_peak: float, turns: float | None = None) -> float:
    """Линейная нагрузка A [А/м] по расточке: ампер-витки всех пазов на длину окружности."""
    n = TURNS_PER_SLOT if turns is None else float(turns)
    return (GEOM.n_slots * n * (float(i_peak) / math.sqrt(2.0))
            / (2.0 * math.pi * GEOM.R_s_out))


def regime_current(name: str = REGIME_DEFAULT) -> float:
    """Амплитуда фазного тока режима из LOAD_REGIMES."""
    return current_for_density(LOAD_REGIMES[name][0])


def sweep_currents() -> list[float]:
    """
    Свип нагрузки: длительный → перегрузочный → пиковый (одинаков обоим материалам, §12).
    Заменяет прежний список [40…150 А], который начинался НА ПИКЕ (33 А/мм²) и уходил до
    125 А/мм², то есть далеко за предел разрушения обмотки.
    """
    j = [LOAD_REGIMES["continuous"][0], 16.0, LOAD_REGIMES["overload"][0], 27.0,
         LOAD_REGIMES["peak"][0]]
    return [round(current_for_density(x), 1) for x in j]


KE_DROP_THRESHOLD = 0.05      # 5% падения K_e = «отказ по стойкости»

# ============================================================ 8. ЧУВСТВИТЕЛЬНОСТЬ (§12, диапазоны)
SENSITIVITY = {
    "h_out": [10.0, 20.0, 50.0],
    "steel_resistivity": [0.42e-6, 0.52e-6],
    "steinmetz_scale": [0.8, 1.0, 1.2],       # ×k_h, k_ex
    "sigma_pm_scale": [0.85, 1.0, 1.15],
    "n_seg": [1, 2, 4],
    "pwm_factor": [1.0, 2.0],                 # sine vs ШИМ×2 на вихревые магнита
}

# ============================================================ builders + оракул
# --- МАГНИТОПРОВОД: 'steel10' = реальное изделие (статор шихт. 0.5 мм, ротор МАССИВНЫЙ);
#     'm270' = электротехническая (эталон сравнения). Данные — sources_registry.md §1, §1a.
STEEL_KINDS = {
    "steel10": dict(curve=steel10_bh_curve, k_th=STEEL10_K_TH, c_vol=STEEL10_DENSITY * STEEL10_CP,
                    resistivity=STEEL10_RESISTIVITY, rotor_solid=True,
                    steinmetz=lambda: SteinmetzCoefficients.steel10_laminated(0.5e-3)),
    # M270 — по datasheet Cogent (НЕ «представительная», которая была в 2.4× мягче реального листа)
    "m270": dict(curve=m270_35a_cogent_bh_curve, k_th=STEEL_K_TH, c_vol=STEEL_C_VOL,
                 resistivity=STEEL_RESISTIVITY, rotor_solid=False,
                 steinmetz=SteinmetzCoefficients.m270_35a_cogent),
}
STEEL_KIND_DEFAULT = "steel10"        # реальный магнитопровод изделия
LAMINATION_THICKNESS = 0.5e-3         # [Sergey] шихтованный статор
ROTOR_MU_R = 500.0                    # μ_r ярма ротора для скин-глубины (⚠ в чувствительность)


def build_thermal_props(material: str, n_rpm: float = N_OPER,
                        steel_kind: str = STEEL_KIND_DEFAULT) -> MachineThermalProperties:
    """Тепловые свойства по регионам: магнит + магнитопровод + k_eff зазора при скорости."""
    m = MAGNET_PROPS[material]
    s = STEEL_KINDS[steel_kind]
    k_gap = airgap_k_eff(n_rpm)
    return MachineThermalProperties(
        k_by_region={
            "air_gap": k_gap, "stator_yoke": s["k_th"], "tooth": s["k_th"],
            "slot": 1.0, "magnet": m["k_th"], "rotor_yoke": s["k_th"],
        },
        c_by_region={
            "air_gap": 1.0e3, "stator_yoke": s["c_vol"], "tooth": s["c_vol"],
            "slot": 2.5e6, "magnet": m["density"] * m["cp"], "rotor_yoke": s["c_vol"],
        },
    )


def steel_loss_kwargs(steel_kind: str = STEEL_KIND_DEFAULT) -> dict:
    """Параметры потерь магнитопровода для `run_machine_thermal_demag` (шихтовка/массив)."""
    s = STEEL_KINDS[steel_kind]
    kw = dict(steinmetz=s["steinmetz"](), rotor_solid=s["rotor_solid"])
    if s["rotor_solid"]:
        kw.update(sigma_rotor=1.0 / s["resistivity"], rotor_mu_r=ROTOR_MU_R)
    return kw


def build_scenario(material: str, steel_kind: str = STEEL_KIND_DEFAULT):
    """Сценарий машины: материал магнита 'ndfeb'|'smco', магнитопровод 'steel10'|'m270'."""
    magnet = (n42sh_magnet if material == "ndfeb" else sm2co17_magnet)((1.0, 0.0, 0.0))
    return machine_scenario(GEOM, magnet, STEEL_KINDS[steel_kind]["curve"]())


def worst_case_gamma(scenario, *, i_peak: float, turns_per_slot: float = TURNS_PER_SLOT,
                     T: float = T_AMB_HOT, n_coarse: int = 12) -> float:
    """
    Истинно ХУДШИЙ (максимально размагничивающий) электрический угол тока γ [рад] для ДАННОГО
    выравнивания ротора: тот, при котором рабочая точка магнита уходит глубже всего во 2-й
    квадрант (минимум H_op — самое отрицательное поле на магните).

    ⚠ Заменяет прежний ХАРДКОД γ=0. Для реального выравнивания (12/14, rotor_angle=0) d-ось НЕ
    на 0°, а ~160–180°; γ=0 оказался НАМАГНИЧИВАЮЩИМ (не worst-case) — расчёт был неконсервативен
    по полю (проверено развёрткой probe_gamma). Это свойство ГЕОМЕТРИИ (слабо зависит от i,T) —
    считается ОДИН раз на прогон и применяется ко всем точкам (§12: одинаковые условия обоим
    материалам; берём худший угол именно по МАГНИТУ, а не по удобству).
    """
    import math

    def demag_H(g: float) -> float:
        sol = scenario.solve(T=T, i_peak=i_peak, gamma_elec=g,
                             turns_per_slot=turns_per_slot, max_iter=80)
        return float(scenario.operating_point(sol).worst_H_op())   # min (самое отрицательное)

    best_g, best_H = 0.0, math.inf
    for k in range(n_coarse):                              # грубый круг 0…2π
        g = 2.0 * math.pi * k / n_coarse
        H = demag_H(g)
        if H < best_H:
            best_H, best_g = H, g
    step = 2.0 * math.pi / n_coarse                        # уточнение вокруг лучшего
    for dg in (-0.5, -0.25, 0.25, 0.5):
        g = best_g + dg * step
        H = demag_H(g)
        if H < best_H:
            best_H, best_g = H, g
    return best_g % (2.0 * math.pi)

def validate_design(verbose: bool = True) -> VerificationReport:
    """
    ПРОВЕРКА ПРАВДОПОДОБИЯ конструкции и режима (P-B0′, добавлено 2026-09-08).

    Зачем. До аудита 2026-09-03 ни одна проверка не ловила, что сценарий идёт в физически
    невозможном режиме: 33,4 А/мм² подписаны «номиналом», а обмотка не позволяла машине
    раскрутиться до заявленных 7000 об/мин ни при какой модуляции. Здесь это закрыто пятью
    проверками, каждая со своей нормой и источником (см. `fem2d/verification.py`).

    ⚠ Строит сетку и решает магнитостатику ⇒ занимает секунды, поэтому вынесена отдельно от
    дешёвой `validate()`.
    """
    # ⚠ Проверка ОБЯЗАНА идти на сетке, разрешающей зазор: на грубой 3 мм коэффициент
    #   проницания занижался с 2,73 до 2,14 — то есть сама проверка врала бы в ту сторону,
    #   в которую мы делаем вывод о конструкции.
    geom = dataclasses.replace(GEOM, mesh_size_by_region=critical_mesh())
    sc = machine_scenario(geom, n42sh_magnet((1.0, 0.0, 0.0)),
                          STEEL_KINDS[STEEL_KIND_DEFAULT]["curve"]())
    sol = sc.solve(T=20.0, i_peak=0.0, gamma_elec=0.0,
                   turns_per_slot=TURNS_PER_SLOT, max_iter=400)
    ke = sc.back_emf_constant(sol, turns_per_slot=TURNS_PER_SLOT)
    op = sc.operating_point(sol)
    pc = float(np.average(op.permeance, weights=op.cell_volume))

    i_cont = regime_current("continuous")
    j_cont = current_density(i_cont)
    a_cont = linear_current_density(i_cont)
    # ⚠ Угол максимального момента ИЩЕТСЯ, а не берётся как π/2: при данном выравнивании
    #   ротора (12/14, rotor_angle=0) q-ось не на 90° — подстановка π/2 даёт момент
    #   ОТРИЦАТЕЛЬНЫЙ. Та же ловушка уже была с γ=0 для d-оси (см. worst_case_gamma).
    torque = max(sc.torque(sc.solve(T=20.0, i_peak=i_cont, gamma_elec=2.0 * math.pi * k / 12,
                                    turns_per_slot=TURNS_PER_SLOT, max_iter=300))
                 for k in range(12))
    v_gap = math.pi * (0.5 * (GEOM.R_s_out + GEOM.R_mag_in)) ** 2 * GEOM.axial_length
    sigma = torque / (2.0 * v_gap)

    report = VerificationReport(checks=[
        check_airgap_resolution(sc.geometry.mesh,
                                sc.geometry.region == int(Region.AIR_GAP), geom.air_gap),
        check_voltage_headroom(ke, N_OPER, U_DC_NOM, MODULATION),
        check_current_density(j_cont, "continuous"),
        check_thermal_loading(a_cont, j_cont * 1e6),
        check_tangential_stress(sigma),
        check_permeance_coefficient(pc),
    ])
    if verbose:
        print("-" * 74)
        print("ПРАВДОПОДОБИЕ КОНСТРУКЦИИ И РЕЖИМА (длительная точка %.1f А, %.1f А/мм²)"
              % (i_cont, j_cont))
        print("  K_e = %.5f В·с/рад ⇒ эквивалентное kV = %.0f (паспорт %.0f, конвенция %s)"
              % (ke, kv_from_ke(ke, KV_CONVENTION), KV, KV_CONVENTION))
        print("  ХХ по потолку шины: %.0f об/мин (%s); рабочая %.0f = %.0f %% от ХХ"
              % (max_speed_rpm(ke, U_DC_NOM, MODULATION), MODULATION, N_OPER,
                 100.0 * N_OPER / max_speed_rpm(ke, U_DC_NOM, MODULATION)))
        print("-" * 74)
        for c in report.checks:
            print("  [%-4s] %-28s %s" % (c.status, c.name, c.value))
            if not c.ok:
                print("         → %s" % c.hint)
        print("-" * 74)
    return report


def validate() -> None:
    """Оракул P-B0: пред-регистрированные условия полны и непротиворечивы."""
    assert 0.0 < DT <= 0.5, "dt должен быть ≤0.5 (калибровка)"
    assert T_CAP > T_AMB_HOT > 20.0, "T_cap > горячая среда > 20°"
    assert H_IN > H_OUT, "статор в раму охлаждается сильнее наружной конвекции"
    assert abs(F_ELEC - 816.7) < 1.0, "частота потерь при 7000 об/мин ≈817 Гц"
    # --- конвенции и режимы (P-B0′, правка аудита 2026-09-03) ---
    assert KV_CONVENTION in KV_CONVENTIONS, "конвенция kV должна быть названа явно"
    assert MODULATION in MODULATION_LIMITS, "тип модуляции должен быть назван явно"
    # витки заданы потолком шины: ЭДС при рабочих оборотах обязана влезать
    ke_model = KE_PER_TURN_MEASURED * TURNS_PER_SLOT
    assert max_speed_rpm(ke_model, U_DC_NOM, MODULATION) >= N_OPER, (
        "обмотка не позволяет достичь %.0f об/мин на шине %.1f В: предел %.0f"
        % (N_OPER, U_DC_NOM, max_speed_rpm(ke_model, U_DC_NOM, MODULATION)))
    # и не должна быть настолько слабой, чтобы паспортное kV потеряло смысл (±20 %)
    assert 0.8 < kv_from_ke(ke_model, KV_CONVENTION) / KV < 1.2, (
        "эквивалентное kV=%.0f разошлось с паспортным %.0f более чем на 20 %%"
        % (kv_from_ke(ke_model, KV_CONVENTION), KV))
    # режимы упорядочены по плотности тока и покрывают длительный/перегруз/пик
    js = [LOAD_REGIMES[k][0] for k in ("continuous", "overload", "peak")]
    assert js == sorted(js), "режимы должны идти по возрастанию плотности тока"
    assert LOAD_REGIMES["continuous"][1] is None, "длительный режим без ограничения времени"
    assert all(LOAD_REGIMES[k][1] is not None for k in ("overload", "peak")), (
        "перегрузочный и пиковый режимы обязаны нести допустимую длительность")
    # k_eff зазора: статика → k_air, растёт с оборотами, на 7000 << старого 0.5
    assert abs(airgap_k_eff(0.0) - K_AIR) < 1e-9, "статика: k_eff=k_air"
    assert airgap_k_eff(7000.0) > airgap_k_eff(1000.0), "k_eff растёт с оборотами"
    assert airgap_k_eff(7000.0) < 0.2, "k_eff@7000 << ad-hoc 0.5 (зазор изолирует)"
    # материалы строятся, регионы покрыты
    for mat in ("ndfeb", "smco"):
        tp = build_thermal_props(mat)
        g = build_scenario(mat).geometry
        k, c = tp.cell_fields(g)
        assert (k > 0).all() and (c > 0).all(), f"{mat}: тепл. свойства покрывают все ячейки"
        assert magnet_conductivity(mat, 20.0) > 0
    # SmCo проводнее NdFeB ⇒ больше вихревых при равном dB/dt (честно)
    assert magnet_conductivity("smco", 20.0) > magnet_conductivity("ndfeb", 20.0)


def _summary() -> str:
    L = ["=" * 74, "P-B0 ПРЕД-РЕГИСТРАЦИЯ сценария Статьи 1 (условия ДО прогонов, §12)", "=" * 74]
    L += [
        "МОТОР: kV=%.0f (конвенция %s), 6S(U_ном=%.1f/полн=%.1f В), P_пик=%.0f Вт"
        % (KV, KV_CONVENTION, U_DC_NOM, U_DC_FULL, P_MAX),
        "  K_e паспорт=%.5f В·с/рад, модель=%.5f (%.0f вит.×%.6f), ХХ по шине=%.0f об/мин"
        % (KE_NAMEPLATE, KE_PER_TURN_MEASURED * TURNS_PER_SLOT, TURNS_PER_SLOT,
           KE_PER_TURN_MEASURED,
           max_speed_rpm(KE_PER_TURN_MEASURED * TURNS_PER_SLOT, U_DC_NOM, MODULATION)),
        "  рабочая n=%.0f → f_эл=%.1f Гц; модуляция %s" % (N_OPER, F_ELEC, MODULATION),
        "ГЕОМ: outrunner 12/14, N=%.0f вит., k_зап=%.2f, γ=0 (worst-case d)" % (TURNS_PER_SLOT, SLOT_FILL),
        "РЕЖИМЫ (по плотности тока): " + "; ".join(
            "%s %.0f А/мм²%s" % (k, v[0], "" if v[1] is None else " (≤%.0f с)" % v[1])
            for k, v in LOAD_REGIMES.items()),
        "СТАЛЬ M270: Штейнмец k_h=%.2e α=%.2f k_c=%.2e k_ex=%.2e; ρ=%.2f µΩм; k=%.0f; c=%.2e" % (
            STEEL_STEINMETZ["k_h"], STEEL_STEINMETZ["alpha"], STEEL_STEINMETZ["k_c"],
            STEEL_STEINMETZ["k_ex"], STEEL_RESISTIVITY * 1e6, STEEL_K_TH, STEEL_C_VOL),
        "МАГНИТ σ@20°: NdFeB=%.2f МС/м, SmCo=%.2f МС/м (SmCo проводнее→больше вихревых)" % (
            magnet_conductivity("ndfeb", 20.0) / 1e6, magnet_conductivity("smco", 20.0) / 1e6),
        "  N_seg=%d [DEFAULT]" % N_SEG,
        "ЗАЗОР k_eff: статика=%.4f, 1740об/мин(порог вихрей), 7000об/мин=%.4f Вт/мК (было 0.5!)" % (
            airgap_k_eff(0.0), airgap_k_eff(7000.0)),
        "ГУ: h_out=%.0f (worst)/%.0f(полёт), h_in=%.0f(→рама), T_frame=%.0f°, T_amb_hot=%.0f° [DEFAULT]" % (
            H_OUT, H_OUT_FLIGHT, H_IN, T_FRAME, T_AMB_HOT),
        "ЧИСЛ: dt=%.2f, T_cap=%.0f°, n_steps=%d, свип I=%s А" % (DT, T_CAP, N_STEPS, sweep_currents()),
        "ЧУВСТВ.: %s" % ", ".join(SENSITIVITY.keys()),
        "=" * 74,
        "[оракул] условия зафиксированы ДО прогонов; физику подключат P-B1..B7.",
        "=" * 74,
    ]
    return "\n".join(L)


if __name__ == "__main__":
    validate()
    print(_summary())
    print("validate(): OK")
    validate_design()
