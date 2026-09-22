from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.rotor_sweep import (
    RotorDamage,
    RotorSweepResult,
    electrical_period_angles,
    sweep_rotor,
)

# ПОТЕРИ В ЖЕЛЕЗЕ (статор) по модели Штейнмеца с РАЗДЕЛЕНИЕМ гистерезис/вихревые.
# Именно ради честного расчёта этих потерь вращение делалось ПЕРЕД железом: имея волну B(θ)
# в неподвижных точках статора за электрический период, берём НАСТОЯЩИЙ размах и производную
# поля по элементу, а не оценку «B_peak из одного снимка × синусоида».
#
# Удельные потери [Вт/кг] на элемент:
#   гистерезис  p_h = k_h · f · B_m^α          (B_m — пик |B| за период)
#   вихревые    p_e = k_e · f² · B_m²           (для СИНУСОИДЫ; калибровка коэффициента)
# Но реальная волна в зубце НЕ синусоида (проходящие зубцы дают гармоники), поэтому вихревые
# считаются из фактической производной: классически P_e ∝ ⟨(dB/dt)²⟩. Через безразмерную
# электрическую фазу θ_e = p·θ_mech = 2πf·t это даёт
#   p_e = 2·k_e·f²·⟨(dB/dθ_e)²⟩,
# что для чистой синусоиды B_m·sinθ_e в точности сводится к k_e·f²·B_m² (⟨cos²⟩=½), а на
# гармониках корректно РАСТЁТ (∝ n² по номеру гармоники) — этого снимок дать не может.
#
# ГРАНИЦА: считается железо СТАТОРА (зубцы+ярмо), стационарное в лаборатории. Ротор в СВОЕЙ
# системе видит квазипостоянное поле (едет вместе с магнитами) ⇒ его потери малы и требуют
# иной трактовки (в системе материала); здесь не считаются. Модель гистерезиса — классическая
# скалярная (по пику |B|); вращательный гистерезис (петля под вращающимся B) не разделяется —
# представительное инженерное приближение. Коэффициенты k_h,α,k_e — ПРЕДСТАВИТЕЛЬНЫЕ, сверять
# с кривой удельных потерь конкретной марки.
#
# ⚠ «По пику |B|» верно только для волны за ОДИН период переменного поля (статор). Для волн с
# постоянной составляющей и для отрезков из нескольких периодов (система ротора) — гистерезис по
# МАЛЫМ ПЕТЛЯМ (`hysteresis_density_minor_loops`, Л-79, 2026-09-11).

_STATOR_IRON = (int(Region.STATOR_YOKE), int(Region.TOOTH))

# --- Сталь 10 (ГОСТ 1050) — магнитопровод реального изделия (статор шихт. 0.5 мм, ротор массив) ---
# Источники и доверие — `docs/sources_registry.md`. Кривая B(H) — `steel_curves.steel10_bh_curve`.
STEEL10_RESISTIVITY = 0.14e-6     # Ом·м, низкоуглеродистая сталь (лит.); в 3.7× проводнее M270
STEEL10_DENSITY = 7856.0          # кг/м³ (ГОСТ 1050)
STEEL10_K_TH = 52.0               # Вт/(м·К) — ВЫШЕ, чем у электротехнической (25): лучше отводит тепло
STEEL10_CP = 470.0                # Дж/(кг·К)
STEEL10_K_HYST_LIT = 0.10         # ⚠ ЛИТЕРАТУРНАЯ оценка гистерезиса [Вт/(кг·Гц·Тл^α)], доверие LOW
STEEL10_ALPHA_LIT = 1.67          # показатель Штейнмеца (как у железа)

# --- Кремнистая электротехническая сталь (M270-35A, datasheet Cogent) -------------------------
# Гистерезисный член — из 2-членной регрессии к таблице удельных потерь Cogent (`m270_35a_cogent`);
# удельное сопротивление и плотность — Cogent/SIJ. Сверено тестом: аналитический вихревой
# коэффициент при 0,35 мм совпадает с фитом Cogent (5,04e-5) — `test_steel10_losses.py`.
M270_K_HYST = 1.83e-2             # Вт/(кг·Гц·Тл^α)
M270_ALPHA = 1.67
M270_RESISTIVITY = 0.52e-6        # Ом·м
M270_DENSITY = 7690.0             # кг/м³


@dataclass(frozen=True, slots=True)
class SteinmetzCoefficients:
    """
    Коэффициенты Штейнмеца [удельные потери в Вт/кг, f в Гц, B в Тл] + плотность стали.

    p = k_hyst·f·B^alpha + k_eddy·f²·B²  (сепарация гистерезис/вихревые).
    """

    k_hyst: float          # Вт·с/(кг·Тл^alpha)
    alpha: float           # показатель Штейнмеца (обычно 1.6…2.2)
    k_eddy: float          # Вт·с²/(кг·Тл²)
    density: float         # плотность стали [кг/м³]

    def __post_init__(self) -> None:
        for nm in ("k_hyst", "k_eddy", "density"):
            if getattr(self, nm) <= 0.0:
                raise ValueError(f"{nm} must be positive.")
        if not (1.0 <= self.alpha <= 3.0):
            raise ValueError("alpha должен быть в [1, 3].")

    @staticmethod
    def m270_35a() -> "SteinmetzCoefficients":
        """
        Представительный M270-35A. ⚠ ОРИЕНТИРОВОЧНО: подобрано под ~2.3 Вт/кг при 1.5 Тл/50 Гц
        (доля вихревых ~15 %) — сверить с кривой удельных потерь марки перед использованием
        в отчёте. Плотность электротехнической стали ≈ 7650 кг/м³.
        """
        return SteinmetzCoefficients(k_hyst=0.017, alpha=2.0, k_eddy=4.3e-5, density=7650.0)

    @staticmethod
    def m270_35a_cogent() -> "SteinmetzCoefficients":
        """
        M270-35A, 2-членный фит к таблице удельных потерь Cogent (P-B3, `bfull_parameters.md`):
        k_h=1.83e-2, α=1.67, k_e=6.9e-5, ρ=7690 (средн. ошибка ~4.4 % на 50–400 Гц/0.2–1.5 Тл).
        Сверка: 1.0 Тл/50 Гц → ~1.09 Вт/кг (Cogent 1.01); 1.5 Тл/50 Гц → ~2.19 (Cogent 2.47).
        Прослеживаемый источник — регрессия к datasheet, не подгонка под результат.
        """
        return SteinmetzCoefficients(k_hyst=1.83e-2, alpha=1.67, k_eddy=6.9e-5, density=7690.0)

    @staticmethod
    def from_lamination(*, k_hyst: float, alpha: float, thickness: float,
                        resistivity: float, density: float) -> "SteinmetzCoefficients":
        """
        Коэффициенты шихтованного листа с ВЫВЕДЕННОЙ АНАЛИТИЧЕСКИ вихревой составляющей:
            k_eddy = π²·d²/(6·ρ·ρ_m)   [Вт/(кг·Гц²·Тл²)]
        (классические вихревые в листе толщиной d: P_v = π²d²f²B²/(6ρ) на объём).
        Это ФИЗИКА, не подгонка: сверка на M270 (d=0.35 мм, ρ=0.52 µΩ·м, ρ_m=7690) даёт
        5.04e-5 — совпадает с фитом к таблице Cogent. Гистерезисный член k_hyst задаётся
        отдельно (из измерений или литературы — из кривой B(H) он невыводим).
        """
        for nm, v in (("thickness", thickness), ("resistivity", resistivity), ("density", density)):
            if v <= 0.0:
                raise ValueError(f"{nm} must be positive.")
        k_eddy = math.pi ** 2 * float(thickness) ** 2 / (6.0 * float(resistivity) * float(density))
        return SteinmetzCoefficients(k_hyst=k_hyst, alpha=alpha, k_eddy=k_eddy, density=density)

    @staticmethod
    def steel10_laminated(thickness: float = 0.5e-3) -> "SteinmetzCoefficients":
        """
        **Сталь 10 (ГОСТ 1050), шихтованный статор** изделия (лист 0.5 мм по умолчанию).

        k_eddy — выведен аналитически из ρ=0.14 µΩ·м и толщины листа (см. `from_lamination`).
        ⚠ k_hyst=0.10, α=1.67 — **ЛИТЕРАТУРНАЯ ОЦЕНКА** (низкоуглеродистая сталь: суммарные
        удельные потери ≈10–13 Вт/кг при 1.5 Тл/50 Гц; за вычетом вихревых ≈2.1 остаётся
        ≈9.5 на гистерезис). Доверие LOW — подлежит замене измерениями и ВХОДИТ В АНАЛИЗ
        ЧУВСТВИТЕЛЬНОСТИ (§12). Коэрцитивность стали 10 (~240–400 А/м) в 6–10 раз выше, чем
        у M270 (~40 А/м), что и объясняет масштаб.

        Следствие для машины: при 817 Гц потери Стали 10 примерно на порядок выше, чем у
        M270-0.35 мм (вихревые ∝ d²/ρ: (0.5/0.35)²·(0.52/0.14) ≈ 7.4×).
        """
        return SteinmetzCoefficients.from_lamination(
            k_hyst=STEEL10_K_HYST_LIT, alpha=STEEL10_ALPHA_LIT, thickness=thickness,
            resistivity=STEEL10_RESISTIVITY, density=STEEL10_DENSITY,
        )

    @staticmethod
    def silicon_steel_laminated(thickness: float = 0.2e-3) -> "SteinmetzCoefficients":
        """
        **Кремнистая электротехническая сталь, шихтованный статор** заданной толщины листа.

        Зачем (сверка с Scorpion IM-8008, 2026-09-10): статор серийного БПЛА-мотора считался
        из Стали 10, и на 439 Гц модель давала потерь в ядре 15,1 Вт при ВСЕХ измеренных
        потерях мотора 10,9 Вт — физически невозможно. Сталь 10 — магнитопровод изделия
        Sergey; серийные моторы делают из кремнистой стали, у которой гистерезис в ~5,5 раза
        ниже (коэрцитивность ~40 против 240–400 А/м).

        Состав коэффициентов:
          · вихревой — ВЫВЕДЕН аналитически из толщины листа и ρ (`from_lamination`): физика;
          · гистерезисный — из регрессии к datasheet M270-35A (Cogent): ПРИНЯТ как свойство
            сплава и перенесён на другую толщину.
        ⚠ Марка стали у покупных моторов не публикуется (у IM-8008 — «0.2mm, Imported»,
          вероятно класс 20JNEH/B20AT). Тонкий лист обычно даёт чуть БОЛЬШИЙ гистерезис на кг
          ⇒ ±20 % — в анализ чувствительности.
        ⚠ Избыточные (аномальные) потери Бертотти в двухчленной модели не выделены: на
          килогерцах они могут быть заметны и занижать потери сверху.
        """
        return SteinmetzCoefficients.from_lamination(
            k_hyst=M270_K_HYST, alpha=M270_ALPHA, thickness=thickness,
            resistivity=M270_RESISTIVITY, density=M270_DENSITY,
        )

    def specific_hysteresis(self, B_peak, freq: float):
        """Удельные гистерезисные потери k_h·f·B_m^α [Вт/кг] (векторизуемо по B_peak)."""
        B = np.asarray(B_peak, dtype=float)
        return self.k_hyst * float(freq) * np.power(np.maximum(B, 0.0), self.alpha)

    def specific_eddy_sinusoid(self, B_peak, freq: float):
        """Удельные вихревые потери СИНУСОИДЫ k_e·f²·B_m² [Вт/кг] (эталон калибровки)."""
        B = np.asarray(B_peak, dtype=float)
        return self.k_eddy * float(freq) ** 2 * B * B


def electrical_frequency(params: OutrunnerPMSMParams, speed_rpm: float) -> float:
    """Электрическая частота f = p·n/60 [Гц] по механической скорости [об/мин]."""
    return (params.n_poles // 2) * float(speed_rpm) / 60.0


def stator_iron_probes(params: OutrunnerPMSMParams):
    """
    Неподвижные пробные точки в железе СТАТОРА (центроиды его ячеек эталонной геометрии)
    и их площади. B в этих точках снимается на каждом угле ротора (см. `sweep_rotor`).
    Возвращает (points:(P,2), areas:(P,)).
    """
    geo = build_outrunner_spm_pmsm(params)
    idx = np.where(np.isin(geo.region, _STATOR_IRON))[0]
    pts = np.array([geo.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
    areas = np.array([geo.mesh.cell_area(int(c)) for c in idx], dtype=float)
    return pts, areas


def eddy_specific_from_waveform(B_series: np.ndarray, freq: float,
                                coeffs: SteinmetzCoefficients) -> np.ndarray:
    """
    Удельные ВИХРЕВЫЕ потери [Вт/кг] по элементу из ВОЛНЫ B(θ_e) за ОДИН электрический период.

    `B_series` — (N, P, 2): B в P точках на N РАВНОМЕРНЫХ углах ровно одного электрического
    периода (θ_e = 0…2π). p_e = 2·k_e·f²·⟨(dB/dθ_e)²⟩, производная — центральной разностью на
    ПЕРИОДИЧЕСКОЙ сетке (обе компоненты B). Захватывает гармоники зубцовой волны.
    """
    B = np.asarray(B_series, dtype=float)
    n = B.shape[0]
    dtheta = 2.0 * math.pi / n
    dB = (np.roll(B, -1, axis=0) - np.roll(B, 1, axis=0)) / (2.0 * dtheta)   # (N,P,2), периодич.
    mean_sq = (dB ** 2).sum(axis=2).mean(axis=0)                             # ⟨(dB/dθ_e)²⟩ по точке
    return 2.0 * coeffs.k_eddy * float(freq) ** 2 * mean_sq


@dataclass(frozen=True, slots=True)
class IronLossResult:
    """Потери в железе статора за электрический период."""

    hysteresis: float             # суммарные гистерезисные потери [Вт]
    eddy: float                   # суммарные вихревые потери [Вт]
    freq: float                   # электрическая частота [Гц]
    iron_mass: float              # масса железа статора [кг]

    @property
    def total(self) -> float:
        return float(self.hysteresis + self.eddy)


def iron_loss_from_probe_waveform(
    B_series: np.ndarray, probe_areas: np.ndarray, *, freq: float,
    axial_length: float, coeffs: SteinmetzCoefficients,
) -> IronLossResult:
    """
    Потери в железе из готовой волны B(θ_e) в пробных точках статора.

    `B_series` — (N,P,2) за один электрический период; `probe_areas` — (P,) площади ячеек.
    Масса элемента = ρ·area·L; удельные потери × масса, суммируются по точкам.
    """
    B = np.asarray(B_series, dtype=float)
    areas = np.asarray(probe_areas, dtype=float).reshape(-1)
    if B.ndim != 3 or B.shape[1] != areas.size or B.shape[2] != 2:
        raise ValueError("B_series должно быть (N, P, 2) с P = число площадей.")

    Bmag = np.hypot(B[:, :, 0], B[:, :, 1])          # |B|(θ) по точкам
    B_peak = Bmag.max(axis=0)                         # пик за период
    p_h = coeffs.specific_hysteresis(B_peak, freq)             # Вт/кг
    p_e = eddy_specific_from_waveform(B, freq, coeffs)         # Вт/кг

    mass = coeffs.density * areas * float(axial_length)        # кг по элементу
    return IronLossResult(
        hysteresis=float((p_h * mass).sum()),
        eddy=float((p_e * mass).sum()),
        freq=float(freq),
        iron_mass=float(mass.sum()),
    )


# --- Гистерезис по МАЛЫМ ПЕТЛЯМ (Л-79, 2026-09-11) ----------------------------------------------
# Энергия петли Штейнмеца на кг — k_h·B_m^α (отсюда p_h = k_h·f·B_m^α для одной петли за период).
# У волны произвольной формы петли выделяются «дождевым потоком» (rainflow): каждая замкнутая
# петля с полуразмахом a даёт k_h·a^α; постоянная составляющая петель НЕ образует.
# Вектор B раскладывается на главные оси его колебаний (собственные векторы ковариации B − ⟨B⟩):
# у переменного поля одного направления вся петля — на большой оси, и результат в точности
# k_h·f·B_m^α (как у статора); вращающееся поле даёт две петли — приближение «эллипса».
# Мощность = f_отр·Σ_петли k_h·a^α, где f_отр — частота повторения снятого отрезка. Отрезок из
# k периодов даёт в k раз больше петель при в k раз меньшей f_отр ⇒ ответ от выбора отрезка НЕ
# зависит. «Пик |B| × f_отр» этим свойством не обладает: на холостом ходу IM-8008 он считал
# постоянный поток кольца ротора полной петлёй и менялся в 9 раз при смене пролёта свипа.
# ⚠ Не учтено: рост потерь малой петли при постоянном подмагничивании (поправки типа Лаверса) —
#   это занижение; порядок величины — в анализ чувствительности.

def rainflow_half_ranges(x) -> tuple[np.ndarray, np.ndarray]:
    """
    Полуразмахи петель ПЕРИОДИЧЕСКОГО сигнала методом «дождевого потока» (трёхточечный,
    ASTM E1049). Сигнал начинается с глобального максимума и замыкается им же — тогда все
    петли полные. Возвращает (полные петли, полупетли); полупетли — остаток вырожденных
    случаев, учитываются с весом ½.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 2:
        return np.zeros(0), np.zeros(0)
    k = int(np.argmax(x))
    y = np.concatenate([x[k:], x[:k], x[k:k + 1]])
    tp = [float(y[0])]                                   # точки поворота
    for v in y[1:]:
        v = float(v)
        if v == tp[-1]:
            continue
        if len(tp) >= 2 and (tp[-1] - tp[-2]) * (v - tp[-1]) > 0.0:
            tp[-1] = v                                   # тот же ход — продолжить
        else:
            tp.append(v)
    stack: list[float] = []
    full: list[float] = []
    for p in tp:
        stack.append(p)
        while len(stack) >= 3:
            x_rng = abs(stack[-1] - stack[-2])
            y_rng = abs(stack[-2] - stack[-3])
            if x_rng < y_rng:
                break
            full.append(0.5 * y_rng)
            del stack[-3:-1]
    half = [0.5 * abs(stack[i + 1] - stack[i]) for i in range(len(stack) - 1)]
    return np.asarray(full, dtype=float), np.asarray(half, dtype=float)


def hysteresis_density_minor_loops(
    B_series: np.ndarray, probe_idx: np.ndarray, n_cells: int, *,
    freq: float, coeffs: SteinmetzCoefficients,
) -> np.ndarray:
    """
    Гистерезисная плотность потерь [Вт/м³] по МАЛЫМ ПЕТЛЯМ волны B (N,P,2).

    `B_series` — волна за снятый отрезок (любое целое число периодов поля); `freq` — частота
    повторения этого отрезка [Гц]. По каждой ячейке: главные оси колебаний B − ⟨B⟩, по каждой
    оси — петли «дождевым потоком», q = ρ·k_h·f·Σ a^α. Ноль вне пробных ячеек.
    """
    B = np.asarray(B_series, dtype=float)
    idx = np.asarray(probe_idx, dtype=int).reshape(-1)
    if B.ndim != 3 or B.shape[1] != idx.size or B.shape[2] != 2:
        raise ValueError("B_series должно быть (N, P, 2) с P = число probe_idx.")
    q = np.zeros(int(n_cells), dtype=float)
    if float(freq) <= 0.0:
        return q
    dB = B - B.mean(axis=0, keepdims=True)
    for j in range(idx.size):
        d = dB[:, j, :]
        _, axes = np.linalg.eigh(d.T @ d)                # главные оси колебаний
        e = 0.0
        for k in range(2):
            full, half = rainflow_half_ranges(d @ axes[:, k])
            e += float(np.sum(full ** coeffs.alpha)) + 0.5 * float(np.sum(half ** coeffs.alpha))
        q[idx[j]] = coeffs.k_hyst * float(freq) * e * coeffs.density
    return q


# --- P-B3: ПОЭЛЕМЕНТНАЯ плотность потерь в стали (источник тепла для связки) ---

def iron_loss_density_from_waveform(
    B_series: np.ndarray, probe_idx: np.ndarray, n_cells: int, *,
    freq: float, coeffs: SteinmetzCoefficients, hysteresis: str = "peak",
) -> np.ndarray:
    """
    Плотность потерь в стали q_Fe [Вт/м³] по ячейкам из волны B в пробных ячейках.

    `B_series` (N,P,2) — волна за снятый отрезок, `freq` — частота его повторения; `probe_idx`
    (P,) — индексы этих ячеек на сетке из n_cells. q = (p_h + p_e)·ρ [Вт/кг·кг/м³ = Вт/м³];
    ноль вне пробных ячеек. Вихревые — из ⟨(dB/dt)²⟩, от длины отрезка не зависят.
    `hysteresis`: "peak" — k_h·f·B_пик^α, только для волны за ОДИН период переменного поля
    (статор); "loops" — по малым петлям (`hysteresis_density_minor_loops`), обязательно для
    волн с постоянной составляющей и отрезков из нескольких периодов (система ротора).
    (Чистая функция — тестируется синтетической волной без прогонки ротора.)
    """
    B = np.asarray(B_series, dtype=float)
    idx = np.asarray(probe_idx, dtype=int).reshape(-1)
    if B.ndim != 3 or B.shape[1] != idx.size or B.shape[2] != 2:
        raise ValueError("B_series должно быть (N, P, 2) с P = число probe_idx.")
    p_e = eddy_specific_from_waveform(B, freq, coeffs)         # Вт/кг
    q = np.zeros(int(n_cells), dtype=float)
    if hysteresis == "peak":
        B_peak = np.hypot(B[:, :, 0], B[:, :, 1]).max(axis=0)
        p_h = coeffs.specific_hysteresis(B_peak, freq)         # Вт/кг
        q[idx] = (p_h + p_e) * coeffs.density                  # Вт/м³
    elif hysteresis == "loops":
        q[idx] = p_e * coeffs.density
        q += hysteresis_density_minor_loops(B, idx, n_cells, freq=freq, coeffs=coeffs)
    else:
        raise ValueError("hysteresis: 'peak' или 'loops'.")
    return q


def stator_iron_loss_density(
    params: OutrunnerPMSMParams,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
    *,
    speed_rpm: float,
    i_peak: float = 0.0,
    gamma_elec: float = 0.0,
    turns_per_slot: float = 0.0,
    T: float = 20.0,
    damage: RotorDamage | None = None,
    coeffs: SteinmetzCoefficients | None = None,
    n_positions: int = 24,
    geometries=None,
    relaxation: float = 0.1,
    max_iter: int = 300,
) -> tuple[np.ndarray, RotorSweepResult]:
    """
    Поэлементная карта потерь в стали СТАТОРА q_Fe [Вт/м³] (n_cells,) как ИСТОЧНИК ТЕПЛА
    для связки (`extra_loss`): прогнать ротор за электрический период, снять волну B в
    ячейках железа статора, применить Штейнмец поэлементно. Ноль вне железа статора.

    Ротор в СВОЕЙ системе видит поле квазипостоянным ⇒ его потери от зубцовых гармоник
    считаются в системе ротора вместе с вихревыми магнита (P-B4), здесь НЕ включаются.
    """
    cf = coeffs or SteinmetzCoefficients.m270_35a_cogent()
    geo = build_outrunner_spm_pmsm(params)
    idx = np.where(np.isin(geo.region, _STATOR_IRON))[0]
    pts = np.array([geo.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
    angles = electrical_period_angles(params, int(n_positions), periods=1)
    sweep = sweep_rotor(
        params, magnet, steel, angles=angles, i_peak=i_peak, gamma_elec=gamma_elec,
        turns_per_slot=turns_per_slot, T=T, damage=damage, no_load=False,
        probe_points=pts, geometries=geometries, relaxation=relaxation, max_iter=max_iter,
    )
    freq = electrical_frequency(params, speed_rpm)
    q = iron_loss_density_from_waveform(sweep.probe_B, idx, geo.mesh.n_cells, freq=freq, coeffs=cf)
    return q, sweep


def efficiency(torque_mean: float, speed_rpm: float,
               copper_loss_w: float, iron_loss_w: float) -> float:
    """
    КПД двигателя η = P_вых / (P_вых + P_медь + P_железо).

    P_вых = |M_ср|·ω_мех, ω_мех = 2π·n/60. Берётся модуль момента (режим двигателя). Механические
    потери (трение/вентиляция) сюда НЕ входят — их можно добавить в сумму потерь отдельно.
    Потери должны быть в тех же ВАТТАХ, что и мощность (не «на единицу осевой длины»).
    """
    if speed_rpm <= 0.0:
        raise ValueError("speed_rpm must be positive.")
    if copper_loss_w < 0.0 or iron_loss_w < 0.0:
        raise ValueError("потери не могут быть отрицательными.")
    omega = 2.0 * math.pi * float(speed_rpm) / 60.0
    p_out = abs(float(torque_mean)) * omega
    p_in = p_out + float(copper_loss_w) + float(iron_loss_w)
    return float(p_out / p_in) if p_in > 0.0 else 0.0


def stator_iron_loss(
    params: OutrunnerPMSMParams,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
    *,
    speed_rpm: float,
    i_peak: float = 0.0,
    gamma_elec: float = 0.0,
    turns_per_slot: float = 0.0,
    T: float = 20.0,
    damage: RotorDamage | None = None,
    coeffs: SteinmetzCoefficients | None = None,
    n_positions: int = 24,
    geometries=None,
    relaxation: float = 0.1,
    max_iter: int = 300,
) -> tuple[IronLossResult, RotorSweepResult]:
    """
    Потери в железе статора: прогнать ротор на один электрический период, снять волну B в
    неподвижных точках статора, применить Штейнмец. Возвращает (потери, результат прогонки —
    там же момент/пульсации, чтобы не гонять сетку дважды).

    Скорость `speed_rpm` задаёт частоту f = p·n/60. `damage` — повреждение магнита, если
    считаем машину ПОСЛЕ перегрузки. Один электрический период достаточен для СИММЕТРИЧНОЙ
    (целой) машины; при несимметричном повреждении картина ротора всё равно p-периодична по
    механике, а частота поля в статоре = электрическая, поэтому период тот же.
    """
    cf = coeffs or SteinmetzCoefficients.m270_35a()
    pts, areas = stator_iron_probes(params)
    angles = electrical_period_angles(params, int(n_positions), periods=1)

    sweep = sweep_rotor(
        params, magnet, steel, angles=angles, i_peak=i_peak, gamma_elec=gamma_elec,
        turns_per_slot=turns_per_slot, T=T, damage=damage, no_load=False,
        probe_points=pts, geometries=geometries, relaxation=relaxation, max_iter=max_iter,
    )
    freq = electrical_frequency(params, speed_rpm)
    loss = iron_loss_from_probe_waveform(
        sweep.probe_B, areas, freq=freq, axial_length=params.axial_length, coeffs=cf
    )
    return loss, sweep
