"""
ВАЛИДАЦИЯ решателя на реальном изделии ДП25-16-7-24 (щёточный ДПТ с ПМ, 16 Вт, 24 В).

ЗАЧЕМ: до сих пор решатель проверялся только синтетическими оракулами (MMS, аналитика,
энергобаланс). ДП25 даёт проверку против СЕРТИФИЦИРОВАННОГО изделия с паспортными данными
(ТУ КМИЖ.524212.006) — это раздел «достоверность результатов» диссертации.

ЦЕЛЬ (выведена из ТУ, паспорт самосогласован — сверено по 3 независимым строкам):
    K_e = (U − I₀·R)/ω₀ = (24 − 0.30·3.9)/(7400·2π/60) = 0.02946 В·с/рад

ГЕОМЕТРИЯ: извлечена из STEP «3D Сборка 1.stp» (T-FLEX), слои сходятся точно в Ø25.
ОБМОТКА (чертёж «Параметры обмотки якоря»): простая ВОЛНОВАЯ, 13 пазов = 13 пластин,
60 эфф. проводников в пазу ⇒ Z=780, 2 параллельные ветви (a=1), скос пакета 11.5°.

МЕТОД: для K_e обмоточная модель НЕ нужна — нужен только поток на полюс из расчёта
холостого хода. Классическая формула коллекторной машины:  K_e = p·Z·Φ/(2π·a)·k_скоса.
Поток на полюс в 2D берётся как разность векторного потенциала между межполюсными осями:
    Φ = L·[A_z(θ₁) − A_z(θ₂)]

Запуск:  PYTHONPATH=<repo> python docs/papers/experiments/validate_dp25.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from magcore.domain.magnet_model import ks25dts240_magnet, n35_magnet, n42sh_magnet
from magcore.domain.steel_curves import m270_35a_cogent_bh_curve, steel10_bh_curve
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.constants import MU0

# --- ПАСПОРТ (ТУ) ---
U, R_ARM, N0_RPM, I0 = 24.0, 3.9, 7400.0, 0.30
KE_TARGET = (U - I0 * R_ARM) / (N0_RPM * 2 * math.pi / 60.0)

# --- ОБМОТКА (чертёж) ---
Z_COND, A_PATHS, P_PAIRS = 780, 1, 2          # 13 пазов × 60; простая волновая ⇒ a=1; 4 полюса
SKEW_MECH_DEG = 11.5

# --- ГЕОМЕТРИЯ (STEP), мм → м ---
GEOM = OutrunnerPMSMParams(
    n_slots=13, n_poles=4,
    R_bore=2.15e-3, h_stator_yoke=1.85e-3, h_tooth=4.532e-3,
    air_gap=0.268e-3, h_magnet=1.90e-3, h_rotor_yoke=1.80e-3,
    tooth_width_frac=0.70,            # оценка по меди: 1.53 мм² на паз при k_зап 0.35–0.45
    magnet_embrace=0.726,             # охват 65.3° из 90° (STEP)
    axial_length=16.0e-3,             # длина пакета (STEP)
    mesh_size=0.25e-3,
    # ⚠ Зазор 0.268 мм — ТОНЬШЕ глобального элемента. При mesh_size=0.30 мм на зазор
    # приходилось <1 элемента, поле в нём не разрешалось и Picard НЕ СХОДИЛСЯ (400 итер.).
    # Нужно ≥3 элемента поперёк зазора ⇒ измельчение именно там.
    mesh_size_by_region={"air_gap": 0.08e-3, "magnet": 0.15e-3, "tooth": 0.18e-3},
)


def skew_factor(mech_deg: float, p_pairs: int) -> float:
    """Коэффициент скоса: k = sin(γ_эл/2)/(γ_эл/2), γ_эл = p·γ_мех."""
    g = math.radians(mech_deg) * p_pairs
    return 1.0 if g == 0 else math.sin(g / 2.0) / (g / 2.0)


def flux_per_pole(geo, a_nodes, *, n_probe=721):
    """
    Поток на полюс [Вб] = L·(A_z^max − A_z^min) по окружности в зазоре.
    Для 2D это точное определение потока между соседними межполюсными осями.
    """
    r = 0.5 * (geo.params.R_s_out + geo.params.R_mag_in)      # середина зазора
    th = np.linspace(0.0, 2 * math.pi, n_probe, endpoint=False)
    pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
    mesh = geo.mesh
    cent = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)])
    vals = np.empty(pts.shape[0])
    for i, p in enumerate(pts):                                # A_z в точке ≈ среднее по узлам ячейки
        c = int(np.argmin(((cent - p) ** 2).sum(axis=1)))
        vals[i] = float(np.mean(a_nodes[list(mesh.cell_vertex_indices(c))]))
    return float(vals.max() - vals.min()) * geo.params.axial_length, vals


def dp25_reluctivity(geo, magnet):
    """
    ν по ячейкам с РАЗНЫМИ сталями по регионам (уточнение Sergey 2026-08-05):
    якорь (ярмо+зубцы) — ЭЛЕКТРОТЕХНИЧЕСКАЯ сталь, корпус (ярмо возврата) — Сталь 10.
    Штатный `machine_reluctivity` берёт одну кривую на всё железо, поэтому здесь — вариант
    для валидации конкретного изделия.
    """
    region = geo.region
    nc = geo.mesh.n_cells
    magnet_mask = region == int(Region.MAGNET)
    arm_idx = np.where(np.isin(region, (int(Region.STATOR_YOKE), int(Region.TOOTH))))[0]
    hous_idx = np.where(region == int(Region.ROTOR_YOKE))[0]
    st_arm, st_hous = m270_35a_cogent_bh_curve(), steel10_bh_curve()
    nu_mag = 1.0 / magnet.mu_rec

    def nu_of_B(B_cells):
        nu = np.ones(nc, dtype=float)
        nu[magnet_mask] = nu_mag
        for idx_, curve in ((arm_idx, st_arm), (hous_idx, st_hous)):
            if idx_.size:
                b = np.hypot(B_cells[idx_, 0], B_cells[idx_, 1])
                nu[idx_] = MU0 * np.array([curve.nu_chord(float(x)) for x in b])
        return nu

    return nu_of_B, nu_of_B(np.zeros((nc, 2))), magnet_mask


def run(magnet_factory, name):
    geo = build_outrunner_spm_pmsm(GEOM)
    space = LagrangeP1Space2D(geo.mesh)
    magnet = magnet_factory((1.0, 0.0, 0.0))
    nu_of_B, nu_init, mask = dp25_reluctivity(geo, magnet)
    idx = np.where(mask)[0]
    nu_br = np.zeros((geo.mesh.n_cells, 2))
    nu_br[idx] = (magnet.Br(20.0) / magnet.mu_rec) * geo.magnet_easy_axis[idx]
    # ω=0.05: ярмо якоря (1.85 мм) глубоко насыщено (⟨B⟩≈1.64 Тл) ⇒ отображение Пикара
    # несжимающее при больших ω. Диагностика: ω=0.5/0.3 осциллируют (невязка 5e-1/2.5e-1),
    # ω=0.15 застревает на 2.9e-2, ω=0.05 сходится за 162 итерации до 9.6e-7.
    em = solve_nonlinear_2d_picard(space, nu_of_B=nu_of_B, nu_init=nu_init,
                                   magnetization=nu_br, relaxation=0.05, max_iter=600, tol=1e-6)
    phi, _ = flux_per_pole(geo, em.a)
    ks = skew_factor(SKEW_MECH_DEG, P_PAIRS)
    ke = P_PAIRS * Z_COND * phi / (2 * math.pi * A_PATHS) * ks
    b_gap = phi / (GEOM.magnet_embrace * (2 * math.pi / GEOM.n_poles)
                   * 0.5 * (GEOM.R_s_out + GEOM.R_mag_in) * GEOM.axial_length)
    print("  %-14s сошлось=%-5s итер=%-4d Ф=%.4f мВб  B_зазора≈%.3f Тл  K_e=%.5f В·с/рад"
          % (name, em.converged, em.n_iterations, phi * 1e3, b_gap, ke))
    return ke, phi, em


print("=" * 92)
print("ВАЛИДАЦИЯ НА ИЗДЕЛИИ ДП25-16-7-24 (щёточный ДПТ, 4 полюса, 13 пазов, Ø25)")
print("=" * 92)
print("ПАСПОРТ (ТУ): U=%.0f В, R=%.1f Ом, n₀=%.0f об/мин, I₀=%.2f А" % (U, R_ARM, N0_RPM, I0))
print("ЦЕЛЬ: K_e = %.5f В·с/рад   (Z=%d, a=%d, p=%d, скос %.1f° → k=%.4f)"
      % (KE_TARGET, Z_COND, A_PATHS, P_PAIRS, SKEW_MECH_DEG, skew_factor(SKEW_MECH_DEG, P_PAIRS)))
geo0 = build_outrunner_spm_pmsm(GEOM)
print("СЕТКА: ячеек=%d, узлов=%d; R: расточка %.2f → якорь %.3f → магнит %.2f…%.2f → Ø%.1f мм"
      % (geo0.mesh.n_cells, geo0.mesh.n_vertices, GEOM.R_bore * 1e3, GEOM.R_s_out * 1e3,
         GEOM.R_mag_in * 1e3, GEOM.R_mag_out * 1e3, GEOM.R_out * 2e3))
print("-" * 92)

print("  (сталь: якорь — ЭЛЕКТРОТЕХНИЧЕСКАЯ (M270 Cogent), корпус — Сталь 10)")
ke_n35, _, _ = run(n35_magnet, "N35 ← ИЗДЕЛИЕ")
ke_n42, _, _ = run(n42sh_magnet, "N42SH (справ.)")
ke_sm, _, _ = run(ks25dts240_magnet, "КС25ДЦ-240")

print("-" * 92)
err = 100 * (ke_n35 - KE_TARGET) / KE_TARGET
print("СВЕРКА С ПАСПОРТОМ — реальная комплектация изделия (N35 + электротехн. сталь якоря):")
print("  расчёт K_e=%.5f  vs  паспорт %.5f  ->  ОТКЛОНЕНИЕ %+.1f %%" % (ke_n35, KE_TARGET, err))
print("  ОБРАТНАЯ ПРОВЕРКА: паспорту соответствует Br = %.3f Тл (у N35 принято 1.19 Тл,"
      " диапазон марки 1.17–1.21)" % (1.19 * KE_TARGET / ke_n35))
print("\nПРОЕКТНОЕ СРАВНЕНИЕ (замена магнита в ДП25, при том же токе):")
print("  N42SH  /N35 = %.3f -> момент %+.1f %%" % (ke_n42 / ke_n35, 100 * (ke_n42 / ke_n35 - 1)))
print("  КС25ДЦ-240/N35 = %.3f -> момент %+.1f %%" % (ke_sm / ke_n35, 100 * (ke_sm / ke_n35 - 1)))
print("=" * 92)
