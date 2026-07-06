from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_curves import (
    DemagnetizationCurveBH,
    _GAUSS_TO_TESLA,
    _KOE_TO_A_PER_M,
    demag_curve_from_datasheet,
)


@dataclass(frozen=True, slots=True)
class AnisotropicBHTMagnet:
    """
    Анизотропный температурно-зависимый магнит B(H,T) (curve-driven, A3).
    См. docs/math/nonlinear_materials.md §4.

    Материал задаётся 4 даташит-параметрами (СИ): Br0 [Тл], |HcB0|, |Hk0|, |HcJ0|
    [А/м] (положительные модули) при T0, плюс ось e, поперечная проницаемость
    mu_perp и два температурных коэффициента (в %/°C):
      alpha_Br — по индукции (Br, HcB);   gamma_Hc — по коэрцитивности (Hk, HcJ).

    Recoil-проницаемость mu_rec = Br0/(mu0*HcB0) — производная, T-НЕЗАВИСИМА.

    Кривая размагничивания строится фабрикой `demag_curve_from_datasheet` (C¹:
    recoil-прямая + касательная парабола). Температура масштабирует ПАРАМЕТРЫ
    (Br,HcB ×b; Hk,HcJ ×h) и кривая ПЕРЕСТРАивается ⇒ наклон recoil сохраняется
    (в отличие от равномерного масштабирования кривой, искажающего mu_rec в b/h раз).

    Необратимость (обобщённая форма, любая кривая): B_r_eff = B^maj_T(H_min) -
    mu0*mu_rec*H_min, где H_min — наименьшее достигнутое H_par (состояние ячейки,
    ведёт решатель). Выше колена B^maj_T = recoil ⇒ B_r_eff = Br(T) (потерь нет).

    Чистый NumPy; методы по T/H_min/B векторизуемы (при скалярной T). Данные марок —
    представительные, сверить с datasheet.
    """

    material_id: str
    name: str
    easy_axis: np.ndarray   # ось лёгкого намагничивания (нормируется к |e|=1)
    Br0: float              # ремнантность при T0 [Тл]
    Hcb0: float             # |H_cB| при T0 [А/м] (> 0)
    Hk0: float              # |H_k| (колено) при T0 [А/м] (> 0)
    Hcj0: float             # |H_cJ| при T0 [А/м] (> 0)
    mu_perp: float          # поперечная проницаемость
    alpha_Br: float = 0.12  # темп. коэфф. по индукции [%/°C] (Br, HcB)
    gamma_Hc: float = 0.6   # темп. коэфф. по коэрцитивности [%/°C] (Hk, HcJ)
    T0: float = 20.0        # опорная температура [°C]

    def __post_init__(self) -> None:
        e = np.asarray(self.easy_axis, dtype=float).reshape(-1)
        if e.shape != (3,):
            raise ValueError("easy_axis must be a 3D vector.")
        n = float(np.linalg.norm(e))
        if not np.isfinite(n) or n <= 0.0:
            raise ValueError("easy_axis must be a non-zero finite 3D vector.")
        object.__setattr__(self, "easy_axis", e / n)
        self.validate()

    def validate(self) -> None:
        if not (np.isfinite(self.Br0) and self.Br0 > 0.0):
            raise ValueError("Br0 must be a positive finite remanence.")
        for nm, v in (("Hcb0", self.Hcb0), ("Hk0", self.Hk0), ("Hcj0", self.Hcj0)):
            if not (np.isfinite(v) and v > 0.0):
                raise ValueError(f"{nm} must be a positive finite magnitude [A/m].")
        if not (self.Hk0 < self.Hcj0):
            raise ValueError("Knee |Hk0| must be below intrinsic coercivity |Hcj0|.")
        if self.mu_perp <= 0.0:
            raise ValueError("mu_perp must be positive.")
        for v in (self.alpha_Br, self.gamma_Hc, self.T0):
            if not np.isfinite(v):
                raise ValueError("Temperature parameters must be finite.")

    # --- релуктивности / тензор (recoil-режим; T-независимы) ---
    @property
    def mu_rec(self) -> float:
        return self.Br0 / (MU0 * self.Hcb0)

    @property
    def nu_parallel(self) -> float:
        return 1.0 / (MU0 * self.mu_rec)

    @property
    def nu_perp(self) -> float:
        return 1.0 / (MU0 * self.mu_perp)

    def nu_tensor(self) -> np.ndarray:
        """Тензор релуктивности nu_mag = nu_perp(I - e e^T) + nu_par e e^T."""
        e = self.easy_axis
        ee = np.outer(e, e)
        return self.nu_perp * (np.eye(3) - ee) + self.nu_parallel * ee

    # --- температурные множители и параметры ---
    def _b(self, T) -> float:
        return 1.0 - self.alpha_Br * (float(T) - self.T0) / 100.0

    def _h(self, T) -> float:
        return 1.0 - self.gamma_Hc * (float(T) - self.T0) / 100.0

    def Br(self, T) -> float:
        return self._b(T) * self.Br0

    def Hcb(self, T) -> float:
        return self._b(T) * self.Hcb0

    def Hk(self, T) -> float:
        """Модуль поля колена |H_k(T)| (> 0)."""
        return self._h(T) * self.Hk0

    def Hcj(self, T) -> float:
        return self._h(T) * self.Hcj0

    def knee_field(self, T) -> float:
        """Поле колена со знаком (< 0)."""
        return -self.Hk(T)

    # --- кривая при температуре ---
    def curve_at(self, T) -> DemagnetizationCurveBH:
        """C¹ кривая размагничивания при T (параметры масштабированы, mu_rec сохранён)."""
        b = self._b(T)
        h = self._h(T)
        if b <= 0.0 or h <= 0.0:
            raise ValueError("Temperature out of model range (Br or Hc scaled to <= 0).")
        return demag_curve_from_datasheet(
            curve_id=f"{self.material_id}@{float(T):g}C",
            name=f"{self.name} @ {float(T):g} C",
            Br=b * self.Br0,
            Hcb=b * self.Hcb0,
            Hk=h * self.Hk0,
            Hcj=h * self.Hcj0,
            temperature_c=float(T),
        )

    def reference_curve(self) -> DemagnetizationCurveBH:
        return self.curve_at(self.T0)

    # --- главная кривая / необратимое состояние ---
    def B_major_parallel(self, H_par, T):
        """Нормальная главная кривая B(H_par) вдоль e при T (векторизуемо по H_par)."""
        curve = self.curve_at(T)
        Hc = np.clip(
            np.asarray(H_par, dtype=float),
            float(curve.H_values[0]),
            float(curve.H_values[-1]),
        )
        return np.interp(Hc, curve.H_values, curve.B_values)

    def effective_Br(self, H_min, T):
        """B_r_eff = B^maj_T(H_min) - mu0*mu_rec*H_min (обобщ. форма; >= потерь нет выше колена)."""
        H = np.asarray(H_min, dtype=float)
        return self.B_major_parallel(H, T) - MU0 * self.mu_rec * H

    def irreversible_loss(self, H_min, T):
        """Необратимая потеря ремнантности dBr = Br(T) - B_r_eff >= 0."""
        return self.Br(T) - self.effective_Br(H_min, T)

    def effective_remanence_vector(self, H_min, T) -> np.ndarray:
        """Вектор эффективной ремнантности B_r_eff * e (FEM-источник A-3)."""
        return float(self.effective_Br(H_min, T)) * self.easy_axis

    # --- рабочая точка / риск ---
    def parallel_field(self, B_vec, T, H_min):
        """Рабочая точка H_par из решённого B: H_par = (B.e - B_r_eff)/(mu0 mu_rec)."""
        B = np.asarray(B_vec, dtype=float)
        B_par = B @ self.easy_axis
        return (B_par - self.effective_Br(H_min, T)) / (MU0 * self.mu_rec)

    def risk_margin(self, H_par, T):
        """Маржа m = H_par - H_k(T); m < 0 => за коленом (необратимое размагничивание)."""
        return np.asarray(H_par, dtype=float) - self.knee_field(T)


def magnet_from_datasheet(
    material_id: str,
    name: str,
    easy_axis,
    Br: float,
    Hcb: float,
    Hk: float,
    Hcj: float,
    mu_perp: float | None = None,
    alpha_Br: float = 0.12,
    gamma_Hc: float = 0.6,
    T0: float = 20.0,
) -> AnisotropicBHTMagnet:
    """Магнит из 4 даташит-параметров (СИ). mu_perp по умолчанию = mu_rec (изотропный recoil)."""
    mu_rec = Br / (MU0 * Hcb)
    if mu_perp is None:
        mu_perp = mu_rec
    return AnisotropicBHTMagnet(
        material_id=material_id,
        name=name,
        easy_axis=np.asarray(easy_axis, dtype=float),
        Br0=Br,
        Hcb0=Hcb,
        Hk0=Hk,
        Hcj0=Hcj,
        mu_perp=mu_perp,
        alpha_Br=alpha_Br,
        gamma_Hc=gamma_Hc,
        T0=T0,
    )


def magnet_from_datasheet_cgs(
    material_id: str,
    name: str,
    easy_axis,
    Br_gauss: float,
    Hcb_kOe: float,
    Hk_kOe: float,
    Hcj_kOe: float,
    mu_perp: float | None = None,
    alpha_Br: float = 0.12,
    gamma_Hc: float = 0.6,
    T0: float = 20.0,
) -> AnisotropicBHTMagnet:
    """Магнит из даташита в CGS (Гс, кЭ). 1 Гс=1e-4 Тл, 1 кЭ=1e6/(4*pi) А/м."""
    return magnet_from_datasheet(
        material_id, name, easy_axis,
        Br=Br_gauss * _GAUSS_TO_TESLA,
        Hcb=Hcb_kOe * _KOE_TO_A_PER_M,
        Hk=Hk_kOe * _KOE_TO_A_PER_M,
        Hcj=Hcj_kOe * _KOE_TO_A_PER_M,
        mu_perp=mu_perp,
        alpha_Br=alpha_Br,
        gamma_Hc=gamma_Hc,
        T0=T0,
    )


def excel_reference_magnet(easy_axis, T0: float = 20.0) -> AnisotropicBHTMagnet:
    """
    Магнит из Excel-шаблона методики (верификационный кейс):
    Br=11114 Гс, HcB=10.4, Hk=15.6, HcJ=28.4 кЭ; alpha_Br=0.12, gamma_Hc=0.465 %/°C.
    ⚠ представительные данные — сверить с datasheet.
    """
    return magnet_from_datasheet_cgs(
        "excel-ref", "Excel reference magnet", easy_axis,
        Br_gauss=11114.0, Hcb_kOe=10.4, Hk_kOe=15.6, Hcj_kOe=28.4,
        alpha_Br=0.12, gamma_Hc=0.465, T0=T0,
    )


def n42sh_magnet(easy_axis, T0: float = 20.0) -> AnisotropicBHTMagnet:
    """
    Представительный N42SH (NdFeB). ⚠ данные приближённые — сверить с datasheet
    (Br, H_cB, колено H_k, H_cJ, темп. коэффициенты).
    """
    return magnet_from_datasheet_cgs(
        "N42SH-representative", "N42SH (representative NdFeB)", easy_axis,
        Br_gauss=12900.0, Hcb_kOe=11.6, Hk_kOe=17.0, Hcj_kOe=20.0,
        alpha_Br=0.115, gamma_Hc=0.55, T0=T0,
    )
