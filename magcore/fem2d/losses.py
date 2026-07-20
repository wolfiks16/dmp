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
