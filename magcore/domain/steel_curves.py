from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0

# Предел |B| (Тл), ниже которого хорда ν(B)=H/B заменяется дифференциальным
# пределом ν_d(0) (во избежание деления на ноль). Первый сегмент проходит через
# начало координат, поэтому при 0<B<=B_1 хорда и так точно равна ν_d(0).
_B_EPS: float = 1.0e-12


@dataclass(frozen=True, slots=True)
class SteelBHCurve:
    """
    Нелинейная ИЗОТРОПНАЯ B–H-кривая электротехнической стали (насыщение).

    Хранит монотонные узлы (H_i, B_i) первого квадранта, начинающиеся в начале
    координат (H_0 = B_0 = 0); H и B строго возрастают. Реакция выдаётся в форме
    РЕЛУКТИВНОСТИ как функции |B| — именно так материал входит в FEM-сборку
    H = ν(|B|)·B (см. docs/math/nonlinear_materials.md §2, §3):

      • хордовая         ν(B)   = H(B)/B   — вдоль B; используется в Picard;
      • дифференциальная ν_d(B) = dH/dB    — касательная; в Newton и тензоре ν_t (§2).

    H(B) — кусочно-линейная интерполяция таблицы, инвертированной B→H.

    Пределы (обеспечивают ограниченность [ν_min, ν_max] и коэрцитивность, §6):
      • B→0:     ν(0) := ν_d(0) = H_1/B_1 (точно — первый сегмент через начало);
      • B>B_max: H(B) = H_max + (B−B_max)/μ0  ⇒  ν_d = 1/μ0,  ν → 1/μ0
                 (глубокое насыщение: дифф. проницаемость → μ0, как у вакуума).

    Физическое ограничение данных: дифф. проницаемость μ_diff = dB/dH ≥ μ0 всюду
    ⇔ наклон таблицы dH/dB ≤ 1/μ0 (проверяется в validate()).

    Чистый NumPy (без scipy). Кривая-агностик: конкретные данные (M270-35A и др.)
    задаются фабриками ниже.
    """

    curve_id: str
    name: str
    H_values: np.ndarray  # А/м, строго возрастает, H_0 = 0
    B_values: np.ndarray  # Тл,  строго возрастает, B_0 = 0
    temperature_c: float | None = None

    def __post_init__(self) -> None:
        H = np.asarray(self.H_values, dtype=float)
        B = np.asarray(self.B_values, dtype=float)
        object.__setattr__(self, "H_values", H)
        object.__setattr__(self, "B_values", B)
        self.validate()

    def validate(self) -> None:
        H = self.H_values
        B = self.B_values

        if H.ndim != 1 or B.ndim != 1:
            raise ValueError("H_values and B_values must be 1D arrays.")
        if len(H) != len(B):
            raise ValueError("H_values and B_values must have equal length.")
        if len(H) < 2:
            raise ValueError("Steel B-H curve must contain at least two points.")
        if not np.isfinite(H).all() or not np.isfinite(B).all():
            raise ValueError("Curve arrays must contain only finite values.")
        if abs(float(H[0])) > 1.0e-9 or abs(float(B[0])) > 1.0e-12:
            raise ValueError("Steel B-H curve must start at the origin (H_0 = B_0 = 0).")
        if np.any(np.diff(H) <= 0.0):
            raise ValueError("H_values must be strictly increasing.")
        if np.any(np.diff(B) <= 0.0):
            raise ValueError("B_values must be strictly increasing.")
        # Physical bound: differential permeability dB/dH >= mu0  <=>  dH/dB <= 1/mu0.
        slopes_dHdB = np.diff(H) / np.diff(B)
        if np.any(slopes_dHdB > (1.0 / MU0) * (1.0 + 1.0e-6)):
            raise ValueError(
                "Table slope dH/dB exceeds 1/mu0 (differential permeability below mu0 is unphysical)."
            )

    @property
    def n_points(self) -> int:
        return len(self.H_values)

    @property
    def B_max(self) -> float:
        return float(self.B_values[-1])

    @property
    def H_max(self) -> float:
        return float(self.H_values[-1])

    @property
    def nu_initial(self) -> float:
        """ν_d(0) = H_1/B_1 — начальная дифф. релуктивность (= хорда при B→0)."""
        return float(self.H_values[1] / self.B_values[1])

    @property
    def nu_saturation(self) -> float:
        """1/μ0 — предел ν и ν_d при глубоком насыщении."""
        return 1.0 / MU0

    def H_of_B(self, B: float) -> float:
        """H как функция |B|: кусочно-линейная в таблице, наклон 1/μ0 за B_max."""
        Bv = abs(float(B))
        if Bv <= self.B_max:
            return float(np.interp(Bv, self.B_values, self.H_values))
        return float(self.H_max + (Bv - self.B_max) / MU0)

    def nu_chord(self, B: float) -> float:
        """Хордовая релуктивность ν(B) = H(B)/B (предел ν_d(0) при B→0)."""
        Bv = abs(float(B))
        if Bv <= _B_EPS:
            return self.nu_initial
        return self.H_of_B(Bv) / Bv

    def nu_differential(self, B: float) -> float:
        """Дифференциальная релуктивность ν_d(B) = dH/dB (1/μ0 в насыщении)."""
        Bv = abs(float(B))
        if Bv >= self.B_max:
            return 1.0 / MU0
        idx = int(np.searchsorted(self.B_values, Bv, side="right") - 1)
        idx = max(0, min(idx, self.n_points - 2))
        dH = self.H_values[idx + 1] - self.H_values[idx]
        dB = self.B_values[idx + 1] - self.B_values[idx]
        return float(dH / dB)

    def nu_pair(self, B: float) -> tuple[float, float]:
        """(ν, ν_d) одним вызовом — для сборки касательного тензора (§2)."""
        return self.nu_chord(B), self.nu_differential(B)

    def mu_r_chord(self, B: float) -> float:
        """Относительная хордовая проницаемость μ_r = 1/(μ0·ν) — диагностика."""
        return 1.0 / (MU0 * self.nu_chord(B))


# Представительная нормальная кривая намагничивания M270-35A
# (нонориентированная электротехническая сталь 0.35 мм).
# ВНИМАНИЕ: данные ПРЕДСТАВИТЕЛЬНЫЕ (типовая форма), НЕ из официального datasheet.
# Перед количественным использованием сверить с Cogent / EN 10106 (M270-35A).
_M270_35A_H = (
    0.0, 30.0, 60.0, 100.0, 150.0, 230.0, 360.0, 600.0,
    1100.0, 2200.0, 5000.0, 12000.0, 30000.0, 80000.0, 160000.0,
)
_M270_35A_B = (
    0.0, 0.15, 0.40, 0.75, 1.00, 1.20, 1.35, 1.48,
    1.58, 1.68, 1.78, 1.88, 1.98, 2.10, 2.21,
)


def m270_35a_bh_curve(temperature_c: float | None = None) -> SteelBHCurve:
    """
    Представительная B–H-кривая стали M270-35A.

    ⚠ Данные приближённые (типовая форма нонориентированной стали 0.35 мм);
    сверить с официальным datasheet (Cogent / EN 10106) перед количественным
    использованием. Класс принимает любую валидную таблицу — замена данных
    тривиальна.
    """
    return SteelBHCurve(
        curve_id="M270-35A-representative",
        name="M270-35A (representative normal B-H curve)",
        H_values=np.asarray(_M270_35A_H, dtype=float),
        B_values=np.asarray(_M270_35A_B, dtype=float),
        temperature_c=temperature_c,
    )


# Кривая намагничивания M270-35A по ОФИЦИАЛЬНОМУ datasheet (Cogent/Surahammars SURA®,
# https://www.tatasteeluk.com/sites/default/files/m270-35a_1.pdf; см. sources_registry.md §1).
# Заменяет «представительную» `_M270_35A_*`, которая оказалась в ~2.4 раза МЯГЧЕ реального
# листа (700 против 1700 А/м при 1.5 Тл) — из-за чего прежние расчёты недооценивали насыщение.
# ⚠ Данные datasheet обрываются на 1.8 Тл, где сталь ЕЩЁ магнитно активна (μ_r≈15). Обрыв
# таблицы там недопустим: за B_max класс экстраполирует наклоном 1/μ₀, и получается СКАЧОК
# проницаемости ×15, на котором Пикар в зубцах (они заходят выше 1.8 Тл) разваливается
# (проверено: карта на такой кривой не сделала ни одного шага). Поэтому таблица продолжена
# ФИЗИЧЕСКИМ хвостом насыщения до μ_r→1 — как это естественно выглядит у измеренной кривой
# Стали 10. Точки хвоста (1.9–2.15 Тл) — ТИПОВЫЕ для нонориентированной стали, НЕ из datasheet.
_M270_COGENT_B = (
    0.0, 0.5, 1.0, 1.2, 1.3, 1.4, 1.5, 1.54, 1.6, 1.65, 1.7, 1.77, 1.8,
    1.9, 2.0, 2.1, 2.15,                     # хвост насыщения (типовой, вне datasheet)
)
_M270_COGENT_H = (
    0.0, 58.0, 112.0, 178.0, 272.0, 596.0, 1700.0, 2500.0, 3880.0, 5000.0, 7160.0, 10000.0, 11600.0,
    20000.0, 40000.0, 80000.0, 115000.0,     # μ_r: 9.5 → 4.0 → 2.0 → 1.14 (плавно к вакууму)
)


def m270_35a_cogent_bh_curve(temperature_c: float | None = None) -> SteelBHCurve:
    """
    B–H-кривая M270-35A по **официальному datasheet Cogent** (а не представительная).

    Использовать для количественных расчётов и для честного сравнения магнитопроводов
    с [`steel10_bh_curve`] — обе кривые тогда одинакового происхождения (datasheet/измерение).
    При 1.5 Тл требует 1700 А/м, что практически совпадает со Сталью 10 (1681 А/м): в рабочем
    диапазоне стали близки, а расходятся ПОТЕРЯМИ и насыщением (Ст10 до 2.25 Тл против 1.8+).
    """
    return SteelBHCurve(
        curve_id="M270-35A-Cogent",
        name="M270-35A (Cogent datasheet B-H curve)",
        H_values=np.asarray(_M270_COGENT_H, dtype=float),
        B_values=np.asarray(_M270_COGENT_B, dtype=float),
        temperature_c=temperature_c,
    )


# --- Сталь 10 (ГОСТ 1050), магнитопровод реального изделия ---
# ИСТОЧНИК: файл характеристики от Sergey (2026-08-03), исходно в СГС.
# ⚠ В исходном файле ЗАГОЛОВОК КОЛОНОК ПЕРЕПУТАН: подписано «H  B», фактически
# колонка 1 = B [Гс], колонка 2 = H [Э]. Расшифровка подтверждена физикой:
# насыщение 2.246 Тл, μ_r нач.≈2755, монотонность, dH/dB ≤ 1/μ₀ (7.94e5 vs 7.96e5).
# Обратная трактовка невозможна (дала бы B=1905 Тл).
# Пересчёт: B[Тл]=B[Гс]/1e4;  H[А/м]=H[Э]·79.5774715.
_STEEL10_B_GS = (
    0.0, 5757.0, 6800.0, 7918.0, 8949.0, 9921.0, 10821.0, 11640.0, 12373.0, 13021.0,
    13586.0, 14074.0, 14494.0, 15171.0, 15451.0, 15955.0, 16455.0, 17019.0, 17679.0,
    18045.0, 18432.0, 18831.0, 19236.0, 19636.0, 20022.0, 20384.0, 20713.0, 21003.0,
    21251.0, 21461.0, 21646.0, 21869.0, 22137.0, 22458.0,
)
_STEEL10_H_OE = (
    0.0, 2.09, 2.5, 3.02, 3.63, 4.365, 5.248, 6.31, 7.586, 9.12,
    10.96, 13.18, 15.85, 22.91, 27.54, 39.8, 57.54, 83.18, 120.23,
    144.5, 173.8, 208.9, 251.2, 301.99, 363.08, 436.5, 524.8, 630.95,
    758.7, 912.0, 1096.5, 1318.3, 1584.9, 1905.0,
)
_OE_TO_A_PER_M = 79.5774715


def steel10_bh_curve(temperature_c: float | None = None) -> SteelBHCurve:
    """
    B–H-кривая **Стали 10** (ГОСТ 1050) — магнитопровод реального изделия.

    Данные заказчика (не представительные): насыщение 2.246 Тл, μ_r нач. ≈2755.
    В рабочем диапазоне практически совпадает с M270 (1.5 Тл при 1681 vs 1700 А/м),
    выше — несёт больше потока (2.0 Тл при 28.6 кА/м; насыщение 2.25 против ≈2.0 Тл).

    ⚠ Отличие Стали 10 от электротехнической — НЕ в этой кривой, а в ПОТЕРЯХ:
    удельное сопротивление ≈0.14 µΩ·м против 0.52 (в ~3.5 раза проводнее) ⇒ вихревые
    потери во столько же раз выше при той же толщине листа; массивные детали (ярмо
    ротора) требуют модели СПЛОШНОГО тела, а не удельных Вт/кг (см. `machines/magnet_loss`).
    """
    return SteelBHCurve(
        curve_id="Steel10-GOST1050",
        name="Сталь 10 (ГОСТ 1050), данные изделия",
        H_values=np.asarray(_STEEL10_H_OE, dtype=float) * _OE_TO_A_PER_M,
        B_values=np.asarray(_STEEL10_B_GS, dtype=float) / 1.0e4,
        temperature_c=temperature_c,
    )
