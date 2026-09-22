"""
P-B2 — эффективная теплопроводность воздушного зазора k_eff(n) (Тейлор–Куэтт).

Перенос тепла через зазор вращающейся машины идёт конвекцией и зависит от скорости:
ниже критического числа Тейлора — чистая кондукция (Nu≈2 при узком зазоре, D_h=2g),
выше — вихри Тейлора усиливают перенос (Nu растёт). Заменяет ad-hoc k_gap=const.

Источник: Howey, Childs, Holmes, "Air-Gap Convection in Rotating Electrical Machines",
IEEE Trans. Ind. Electron. 2012, DOI 10.1109/TIE.2010.2100337 (корреляции Bjorklund&Kays,
Becker&Kaye; критич. Ta_m=41.19). Конвенция: Nu = h·D_h/k, D_h=2g ⇒ k_eff = Nu·k_air/2
(при Nu=2 даёт k_eff=k_air — согласованность с кондукцией).
"""
from __future__ import annotations

import math

# Свойства воздуха @~90 °C (Incropera); переопределяемы аргументами.
K_AIR_DEFAULT = 0.030          # Вт/(м·К)
NU_AIR_DEFAULT = 2.1e-5        # м²/с
TA_CRIT = 41.19                # критич. число Тейлора (узкий зазор, неподв. внешний цил.)
TA_CRIT_SQ = TA_CRIT ** 2      # ≈1697 — порог по x = (Ta/F_g)²


def taylor_number(n_rpm: float, r_mean: float, gap: float, nu_air: float = NU_AIR_DEFAULT) -> float:
    """Число Тейлора Ta_m = Ω·√r_m·g^1.5/ν (Howey eq.8). Ω=2π·n/60 [рад/с]."""
    omega = 2.0 * math.pi * float(n_rpm) / 60.0
    return omega * math.sqrt(float(r_mean)) * float(gap) ** 1.5 / float(nu_air)


def airgap_nusselt(Ta: float, F_g: float = 1.04) -> float:
    """
    Число Нуссельта зазора по режиму (селектор x = (Ta/F_g)²):
      x < 1700           → Nu = 2                       (кондукция, узкий зазор)
      1700 ≤ x < 1e4     → Nu = 0.128·x^0.367           (ламинарные вихри, Becker&Kaye)
      x ≥ 1e4            → Nu = 0.409·x^0.241            (турбулентные вихри, Becker&Kaye)
    Корреляции непрерывны на стыках 1700 и 1e4.
    """
    x = (float(Ta) / float(F_g)) ** 2
    if x < TA_CRIT_SQ:
        return 2.0
    if x < 1.0e4:
        return 0.128 * x ** 0.367
    return 0.409 * x ** 0.241


def airgap_k_eff(
    n_rpm: float, r_mean: float, gap: float, *,
    k_air: float = K_AIR_DEFAULT, nu_air: float = NU_AIR_DEFAULT, F_g: float = 1.04,
) -> float:
    """Эффективная теплопроводность зазора k_eff = Nu(Ta)·k_air/2 [Вт/(м·К)]."""
    Ta = taylor_number(n_rpm, r_mean, gap, nu_air)
    return airgap_nusselt(Ta, F_g) * float(k_air) / 2.0


def taylor_critical_speed(r_mean: float, gap: float, nu_air: float = NU_AIR_DEFAULT,
                          F_g: float = 1.04) -> float:
    """Скорость [об/мин] появления вихрей Тейлора (Ta_m = Ta_crit·F_g ⇔ x = Ta_crit²)."""
    omega_cr = TA_CRIT * float(F_g) * float(nu_air) / (math.sqrt(float(r_mean)) * float(gap) ** 1.5)
    return omega_cr * 60.0 / (2.0 * math.pi)
