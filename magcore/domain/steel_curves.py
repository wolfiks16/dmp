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
