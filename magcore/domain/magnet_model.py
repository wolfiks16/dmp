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

    def temperature_limit(self) -> float:
        """
        Верхняя температура валидности модели [°C]: где множитель _b(T) или _h(T)
        обращается в 0 (Br или Hc масштабируется в ≤0). Выше неё магнит вне диапазона
        модели (перегрет/разрушен) — `curve_at` бросит исключение. = T0 + 100/max(коэфф.).
        """
        slopes = [s for s in (self.alpha_Br, self.gamma_Hc) if s > 0.0]
        if not slopes:
            return float("inf")
        return self.T0 + 100.0 / max(slopes)

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
        """
        Нормальная главная кривая B(H_par) вдоль e при T (векторизуемо по H_par).

        Таблица кривой покрывает H ∈ [−H_cJ, 0]. За её краями — то же доопределение, что в ядре
        К6′ (`fem2d.coupled_transient._curve_eval`), а не обрезка: при H > 0 (подмагничивающее
        поле) — линия возврата B_r(T) + μ0·μ_rec·H (потерь нет); левее −H_cJ — консервативно
        полная потеря, линия возврата из начала координат μ0·μ_rec·H (B_r,eff = 0). Обрезка при
        H > 0 давала ложную необратимую «потерю» μ0·μ_rec·H — 0,14 Тл при +100 кА/м (Л-92).
        """
        curve = self.curve_at(T)
        Hv, Bv = curve.H_values, curve.B_values
        H = np.asarray(H_par, dtype=float)
        B = np.interp(np.clip(H, Hv[0], Hv[-1]), Hv, Bv)
        B = np.where(H > Hv[-1], Bv[-1] + MU0 * self.mu_rec * (H - Hv[-1]), B)
        return np.where(H < Hv[0], MU0 * self.mu_rec * H, B)

    def B_major_slope(self, H_par, T):
        """Наклон главной кривой dB/dH_par [Тл·м/А] при T — касательная для метода Ньютона."""
        curve = self.curve_at(T)
        Hv, Bv = curve.H_values, curve.B_values
        H = np.asarray(H_par, dtype=float)
        idx = np.clip(np.searchsorted(Hv, H, side="right") - 1, 0, Hv.size - 2)
        s = (Bv[idx + 1] - Bv[idx]) / (Hv[idx + 1] - Hv[idx])
        return np.where((H >= Hv[-1]) | (H < Hv[0]), MU0 * self.mu_rec, s)

    def effective_Br(self, H_min, T):
        """B_r_eff = B^maj_T(H_min) - mu0*mu_rec*H_min (обобщ. форма; >= потерь нет выше колена)."""
        H = np.asarray(H_min, dtype=float)
        return self.B_major_parallel(H, T) - MU0 * self.mu_rec * H

    def irreversible_loss(self, H_min, T):
        """Необратимая потеря ремнантности dBr = Br(T) - B_r_eff >= 0."""
        return self.Br(T) - self.effective_Br(H_min, T)

    # --- необратимая память: сохранённая доля ремнантности r (постановка (S2), Л-100) ---
    def retention_now(self, H_par, T):
        """
        Доля ремнантности r_now(H∥, T) ∈ [0, 1], которую оставило бы наихудшее поле H∥ при температуре T:
        r_now = clip((B^maj(H∥, T) − μ0·μ_rec·H∥) / B_r(T), 0, 1) — пересечение линии возврата из точки
        главной кривой с осью H = 0, отнесённое к номиналу. Выше колена и при H∥ > 0 — ровно 1: шум
        интерполяции 1 − O(1e-16) привязывается к единице, иначе защёлка копила бы его как «потерю»;
        ниже −H_cJ — 0 (полная потеря). Правило то же, что в ядре К6′
        (`fem2d.coupled_transient.IrreversibleMagnetState.__call__`); совпадение закреплено тестом.

        Переменная состояния необратимости — доля r = min по истории r_now (постановка (S2),
        docs/math/coupled_problem.md): при смене температуры потерянная доля сохраняется, при остывании
        возвращается только обратимая часть через B_r(T). Хранить вместо неё наихудшее поле нельзя —
        при остывании колено уходит глубже, и магнит «вылечился» бы (Л-100).
        """
        r = np.clip(np.asarray(self.effective_Br(H_par, T), dtype=float) / self.Br(T), 0.0, 1.0)
        return np.where(r > 1.0 - 1.0e-9, 1.0, r)

    def switch_field(self, retention, T):
        """
        Поле переключения H*(r, T) [А/м] ячейки с сохранённой долей r: при H∥ ≥ H* закон — линия
        возврата r·B_r(T) + μ0·μ_rec·H∥, при H∥ < H* — главная кривая (новая потеря). Обращение
        r_now(H, T), которое на отрезке таблицы [−H_cJ, H_k] кусочно-линейно и строго возрастает:
        r = 1 → колено H_k(T); r ниже r_now(−H_cJ, T) (такая доля бывает, если потеря получена при
        другой температуре) → −H_cJ(T): линия возврата на всей таблице ниже главной кривой, и новая
        потеря начинается только за −H_cJ. Нужна для коэнергии закона (энергия и виртуальная работа).
        """
        curve = self.curve_at(T)
        Hv = np.asarray(curve.H_values, dtype=float)
        Bv = np.asarray(curve.B_values, dtype=float)
        k = int(np.argmin(np.abs(Hv - self.knee_field(T))))                 # узел колена
        g = (Bv[: k + 1] - MU0 * self.mu_rec * Hv[: k + 1]) / self.Br(T)    # r_now на узлах [−H_cJ, H_k]
        if not np.all(np.diff(g) > 0.0):
            raise ValueError("r_now на отрезке [−H_cJ, H_k] не возрастает строго — таблица кривой некорректна.")
        return np.interp(np.asarray(retention, dtype=float), g, Hv[: k + 1])

    # --- рабочая ветвь закона: одна реализация на 2D и 3D (Л-92, Л-100) ---
    def branch_parallel(self, H_par, T, retention=1.0):
        """
        Закон вдоль лёгкой оси при сохранённой доле ремнантности r: B∥(H∥) и наклон dB∥/dH∥.

        Две ветви гистерезисного оператора: идёт НОВАЯ необратимая потеря (r_now(H∥, T) < r) —
        главная кривая B^maj; иначе линия возврата r·B_r(T) + μ₀·μ_rec·H∥. Ветви сходятся там,
        где r_now = r (поле переключения `switch_field`), поэтому закон непрерывен, а наклон
        берётся у той же ветви, по которой считается B (согласованная линеаризация, Л-21/Л-93).
        r = 1 — новый магнит (главная кривая ниже колена, линия возврата выше).
        """
        H = np.asarray(H_par, dtype=float)
        r = np.asarray(retention, dtype=float)
        mu_rec_abs = MU0 * self.mu_rec
        new_loss = np.asarray(self.retention_now(H, T), dtype=float) < r
        b = np.where(new_loss, self.B_major_parallel(H, T), r * self.Br(T) + mu_rec_abs * H)
        s = np.where(new_loss, self.B_major_slope(H, T), mu_rec_abs)
        return np.asarray(b, dtype=float), np.asarray(s, dtype=float)

    def branch_parallel_inverse(self, B_par, T, retention=1.0):
        """
        Обращение `branch_parallel`: H∥(B∥) и тот же наклон dB∥/dH∥.

        Нужно там, где неизвестное — ИНДУКЦИЯ (планарная постановка через A_z): закон строго
        возрастает по H∥, поэтому обращение однозначно. Ниже точки переключения B* = B∥(H*)
        работает главная кривая (обращается по её таблице), выше — линия возврата. За краями
        таблицы — те же доопределения, что в `B_major_parallel`: выше B_r(T) линия возврата
        вверх, ниже левого края — линия μ₀·μ_rec·H (полная потеря).
        """
        B = np.asarray(B_par, dtype=float)
        r = np.asarray(retention, dtype=float)
        mu_rec_abs = MU0 * self.mu_rec
        curve = self.curve_at(T)
        Hv = np.asarray(curve.H_values, dtype=float)
        Bv = np.asarray(curve.B_values, dtype=float)
        if not np.all(np.diff(Bv) > 0.0):
            raise ValueError("главная кривая должна строго возрастать по B — таблица некорректна.")
        h_star = np.asarray(self.switch_field(r, T), dtype=float)
        b_star = r * self.Br(T) + mu_rec_abs * h_star                       # точка переключения ветвей
        # главная кривая: обращение по тому же отрезку таблицы, по которому интерполируется B
        i = np.clip(np.searchsorted(Bv, B, side="right") - 1, 0, Bv.size - 2)
        s_tab = (Bv[i + 1] - Bv[i]) / (Hv[i + 1] - Hv[i])
        h_maj = Hv[i] + (B - Bv[i]) / s_tab
        above = B > Bv[-1]                                                  # подмагничивание: вверх по возврату
        h_maj = np.where(above, Hv[-1] + (B - Bv[-1]) / mu_rec_abs, h_maj)
        below = B < Bv[0]                                                   # ниже −H_cJ: полная потеря
        h_maj = np.where(below, np.minimum(B / mu_rec_abs, Hv[0]), h_maj)
        s_maj = np.where(above | below, mu_rec_abs, s_tab)
        recoil = B >= b_star
        h = np.where(recoil, (B - r * self.Br(T)) / mu_rec_abs, h_maj)
        s = np.where(recoil, mu_rec_abs, s_maj)
        return np.asarray(h, dtype=float), np.asarray(s, dtype=float)

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


def sm2co17_magnet(easy_axis, T0: float = 20.0) -> AnisotropicBHTMagnet:
    """
    Представительный Sm2Co17 (тип КС25ДЦ, отечественное производство) — магнит для
    ТЕПЛОНАГРУЖЕННЫХ двигателей БПЛА. Ключевое отличие от NdFeB — высокая
    температурная стабильность: |alpha_Br|≈0.03 и |gamma_Hc|≈0.20 %/°C (против
    0.115 и 0.55 у NdFeB) + высокая собственная коэрцитивность ⇒ колено «держится»
    при нагреве. ⚠ данные ПРЕДСТАВИТЕЛЬНЫЕ (ориентир ГОСТ 21559-76 КС25ДЦ: Br 0.90–1.10 Тл,
    HcB 690–780 кА/м) — сверить с datasheet/собственными измерениями производителя.
    """
    return magnet_from_datasheet(
        "KS25DС-representative", "Sm2Co17 (representative КС25ДЦ)", easy_axis,
        Br=1.05, Hcb=780.0e3, Hk=1100.0e3, Hcj=1600.0e3,
        alpha_Br=0.030, gamma_Hc=0.20, T0=T0,
    )


def n35_magnet(easy_axis, T0: float = 20.0) -> AnisotropicBHTMagnet:
    """
    **N35** (NdFeB, СТАНДАРТНАЯ марка без термо-суффикса) — магниты изделия ДП25
    (уточнено Sergey, 2026-08-05).

    Br 1.17–1.21 Тл (взято 1.19), HcB ≥ 868 кА/м (10.9 кЭ), HcJ ≥ 955 кА/м (12 кЭ),
    (BH)max 263–287 кДж/м³. ⚠ В отличие от N42SH (суффикс SH = высокотемпературная),
    у стандартной N-марки **колено близко к HcB** и температурные коэффициенты хуже:
    α_Br ≈ −0.12 %/°C, γ_HcJ ≈ −0.60 %/°C, предельная рабочая T ≈ 80 °C.
    """
    return magnet_from_datasheet_cgs(
        "N35", "N35 (стандартный NdFeB)", easy_axis,
        Br_gauss=11900.0, Hcb_kOe=10.9, Hk_kOe=11.4, Hcj_kOe=12.0,
        alpha_Br=0.12, gamma_Hc=0.60, T0=T0,
    )


def ks25dts240_magnet(easy_axis, T0: float = 20.0) -> AnisotropicBHTMagnet:
    """
    **КС25ДЦ-240** (Sm2Co17, ГОСТ 21559-76) — ВЕРХНЯЯ марка диапазона, конкретно заданная
    для расчётов (Sergey, 2026-08-03), в отличие от `sm2co17_magnet` (середина диапазона).

    Индекс 240 = (BH)max в кДж/м³. Для линейного магнита (BH)max ≈ Br²/(4μ₀) ⇒
    Br = √(4μ₀·240е3) ≈ **1.10 Тл** — верх диапазона ГОСТ (0.90–1.10 Тл); HcB берётся
    верхним по ГОСТ (780 кА/м). Температурные коэффициенты — как у Sm2Co17.
    ⚠ Hk/HcJ — представительные для 2:17 (ГОСТ их не нормирует), сверить с паспортом партии.
    """
    return magnet_from_datasheet(
        "KS25DTs-240", "Sm2Co17 КС25ДЦ-240 (ГОСТ 21559-76)", easy_axis,
        Br=1.10, Hcb=780.0e3, Hk=1150.0e3, Hcj=1600.0e3,
        alpha_Br=0.030, gamma_Hc=0.20, T0=T0,
    )
