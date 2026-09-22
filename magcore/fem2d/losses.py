from __future__ import annotations

import numpy as np

# S3, инкремент 2: ПЛОТНОСТЬ ПОТЕРЬ для источника тепла связанного магнитотеплового расчёта.
# Основной (управляющий runaway) — джоулевы потери меди при ЗАДАННОМ токе: J фиксирована,
# а удельное сопротивление растёт с температурой ρ(T)=ρ0·(1+α(T−T0)) ⇒ q=ρ(T)·J² растёт с T.
# Это и есть положительная обратная связь ядра К6′: чувствительность s=dq/dT=ρ0·α·J²
# (подаётся в runaway_threshold/связку). Железо (гистерезис+вихревые) — отдельно (нужна
# частота/вращение), добавляется позже.

# Медь: ρ0 при 20°C [Ом·м], темп. коэфф. α [1/°C]. Представительные (сверить с маркой).
CU_RHO0 = 1.724e-8
CU_ALPHA = 3.9e-3
CU_T0 = 20.0


def copper_resistivity(T, *, rho0: float = CU_RHO0, alpha: float = CU_ALPHA, T0: float = CU_T0):
    """Удельное сопротивление меди ρ(T)=ρ0·(1+α(T−T0)) [Ом·м] (векторизуемо по T)."""
    return rho0 * (1.0 + alpha * (np.asarray(T, dtype=float) - T0))


def copper_loss_density(j_cells, T_cells, *, rho0: float = CU_RHO0,
                        alpha: float = CU_ALPHA, T0: float = CU_T0) -> np.ndarray:
    """
    Плотность джоулевых потерь меди q = ρ(T)·J² [Вт/м³] по ячейкам. `j_cells` — плотность
    тока [А/м²] (0 вне проводника), `T_cells` — температура ячеек [°C]. Ноль там, где J=0.
    """
    j = np.asarray(j_cells, dtype=float)
    return copper_resistivity(T_cells, rho0=rho0, alpha=alpha, T0=T0) * j * j


def copper_loss_sensitivity(j_cells, *, rho0: float = CU_RHO0, alpha: float = CU_ALPHA) -> np.ndarray:
    """
    Чувствительность потерь dq/dT = ρ0·α·J² [Вт/(м³·K)] по ячейкам — линейная обратная связь
    для порога runaway (runaway.runaway_threshold сравнивает с s_crit) и линеаризованной связки.
    """
    j = np.asarray(j_cells, dtype=float)
    return rho0 * alpha * j * j


# ------------------------------------------------------------------------------------------
# ПЕРЕМЕННЫЕ ПОТЕРИ В МЕДИ ПАЗА: вытеснение тока и эффект близости (модель Дауэлла)
#
# Зачем (сверка с IM-8008, 2026-09-10): на 1000–1300 Гц модель недосчитывала потерь до
# 130 Вт, причём недостача росла круче любой степени частоты — а у БПЛА-моторов электрическая
# частота как раз сотни–тысячи герц. Постоянное сопротивление ρ·L/S здесь занижает потери:
# поле рассеяния паза наводит в каждом слое проводников вихревые токи, и чем выше слой, тем
# сильнее (слои ниже уже создали своё поле). Классическая модель — Dowell P.L., «Effects of
# eddy currents in transformer windings», Proc. IEE 113(8):1387–1394, 1966; для машин её
# применяют к проводникам в пазу, где поле рассеяния одномерно поперёк паза.
#
#   F_R = R_ac/R_dc = Δ·[ ς₁(Δ) + (2/3)(m² − 1)·ς₂(Δ) ],
#   ς₁ = (sh 2Δ + sin 2Δ)/(ch 2Δ − cos 2Δ),   ς₂ = (sh Δ − sin Δ)/(ch Δ + cos Δ),
#   Δ = h/δ·√η,   h = (√π/2)·d (круглый провод → эквивалентная фольга),
#   δ = √(ρ/(π·f·μ₀)) — глубина проникновения, m — число слоёв поперёк поля, η — пористость слоя.
# Малый Δ: F_R ≈ 1 + (5m² − 1)/45·Δ⁴ — именно поэтому потери растут КРУЧЕ f²
# (Δ ∝ √f ⇒ добавка ∝ f², и она умножается на I², который у винта сам растёт как n²).
#
# ⚠ Применяется ТОЛЬКО к части сопротивления, лежащей В ПАЗУ: лобовые части находятся в
#   воздухе, поле рассеяния там на порядок слабее, для них F_R ≈ 1.
# ⚠ Входы (диаметр жилы, число жил, число слоёв) у покупных моторов не публикуются. Пока их
#   нет из обмера, результат выдаётся ДИАПАЗОНОМ по правдоподобной намотке, а не числом.

def copper_skin_depth(freq: float, T: float = CU_T0, *, mu_r: float = 1.0) -> float:
    """Глубина проникновения в меди δ = √(ρ(T)/(π·f·μ₀μ_r)) [м]."""
    from magcore.constants import MU0

    if not (freq > 0.0):
        raise ValueError("freq must be positive.")
    return float(np.sqrt(float(copper_resistivity(T)) / (np.pi * float(freq) * MU0 * mu_r)))


def dowell_ac_factor(delta, layers: int):
    """
    Коэффициент увеличения сопротивления F_R = R_ac/R_dc по Дауэллу (векторизуемо по Δ).

    `delta` — приведённая толщина Δ (≥ 0), `layers` — число слоёв m (≥ 1) поперёк поля.
    Малые Δ считаются по разложению 1 + (5m²−1)/45·Δ⁴ (там точная формула теряет знаки на
    вычитании), большие — с ς₁, ς₂ → 1 (там ch/sh переполняются).
    """
    m = int(layers)
    if m < 1:
        raise ValueError("layers must be >= 1.")
    d = np.asarray(delta, dtype=float)
    if np.any(d < 0.0):
        raise ValueError("delta must be non-negative.")
    out = np.empty_like(d)
    small = d < 1.0e-2
    big = d > 30.0
    mid = ~(small | big)
    out[small] = 1.0 + (5.0 * m * m - 1.0) / 45.0 * d[small] ** 4
    x = d[mid]
    s1 = (np.sinh(2 * x) + np.sin(2 * x)) / (np.cosh(2 * x) - np.cos(2 * x))
    s2 = (np.sinh(x) - np.sin(x)) / (np.cosh(x) + np.cos(x))
    out[mid] = x * (s1 + (2.0 / 3.0) * (m * m - 1.0) * s2)
    out[big] = d[big] * (1.0 + (2.0 / 3.0) * (m * m - 1.0))
    return out if out.ndim else float(out)


def round_wire_delta(*, strand_diameter: float, freq: float, T: float = CU_T0,
                     porosity: float = 0.85) -> float:
    """Приведённая толщина Δ для круглой жилы диаметра d: h = (√π/2)·d, Δ = h/δ·√η."""
    if not (strand_diameter > 0.0 and 0.0 < porosity <= 1.0):
        raise ValueError("strand_diameter > 0 and porosity in (0, 1] required.")
    h = 0.5 * np.sqrt(np.pi) * float(strand_diameter)
    return float(h / copper_skin_depth(freq, T) * np.sqrt(porosity))
